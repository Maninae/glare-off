"""Change-guard validation metrics, the best-checkpoint gate, and which weights `train` exports."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet
from glare_model.train import select_model_to_export
from glare_model.training.checkpointing import BEST_CHECKPOINT_NAME, save_checkpoint_atomically
from glare_model.training.validation_metrics import ValidationMetricAccumulator, compute_checkpoint_selection_score

MAX_CLEAN_SAMPLE_CHANGE = 0.5 / 255


def make_validation_batch() -> dict[str, torch.Tensor]:
    """Sample 0 is glare-free; sample 1 has glare inside the lens and a bloom spot just outside it."""
    clean_eye_crop = torch.full((2, 3, 16, 32), 0.4)
    lens_mask = torch.zeros(2, 1, 16, 32)
    lens_mask[..., 4:12, 4:28] = 1.0
    glare_mask = torch.zeros_like(lens_mask)
    glare_mask[1, :, 5:9, 6:12] = 1.0
    glare_mask[1, :, 0:2, 6:12] = 1.0  # outside the lens
    return {
        "clean_eye_crop": clean_eye_crop,
        "glare_eye_crop": clean_eye_crop + 0.3 * glare_mask,
        "glare_mask": glare_mask,
        "lost_detail_mask": torch.zeros_like(glare_mask),
        "lens_mask": lens_mask,
    }


def summarize_one_batch(blended_crop: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    accumulator = ValidationMetricAccumulator()
    predicted_masks = torch.cat([batch["glare_mask"], batch["lost_detail_mask"]], dim=1)
    accumulator.update(blended_crop, predicted_masks, batch)
    return accumulator.summarize()


def test_change_metrics_measure_glare_free_and_outside_lens_edits():
    batch = make_validation_batch()
    blended_crop = batch["clean_eye_crop"].clone()  # glare removed everywhere, including the bloom spot
    blended_crop[0, :, 0:2, 0:4] += 0.02  # a spurious edit on the glare-free sample, outside the lens
    metrics = summarize_one_batch(blended_crop, batch)
    assert metrics["clean_sample_change"] == pytest.approx(0.02 * 8 / (16 * 32), rel=1e-4)
    assert metrics["clean_sample_change_max"] == pytest.approx(0.02, rel=1e-4)
    # Removing the out-of-lens bloom on sample 1 does not count; only the spurious edit does.
    outside_weight = (1 - batch["lens_mask"]) * (1 - batch["glare_mask"])
    assert metrics["outside_lens_change"] == pytest.approx(0.02 * 8 * 3 / (float(outside_weight.sum()) * 3), rel=1e-4)


def test_clean_sample_change_is_nan_without_glare_free_samples():
    batch = make_validation_batch()
    batch = {key: value[1:] for key, value in batch.items()}
    metrics = summarize_one_batch(batch["clean_eye_crop"], batch)
    assert math.isnan(metrics["clean_sample_change"]) and math.isnan(metrics["clean_sample_change_max"])


def test_selection_rejects_checkpoints_that_edit_clean_samples():
    passing = {"psnr_lens": 30.0, "clean_sample_change": 0.1 / 255}
    failing = {"psnr_lens": 35.0, "clean_sample_change": 2.0 / 255}
    unmeasurable = {"psnr_lens": 28.0, "clean_sample_change": float("nan")}
    assert compute_checkpoint_selection_score(passing, MAX_CLEAN_SAMPLE_CHANGE) == 30.0
    assert compute_checkpoint_selection_score(failing, MAX_CLEAN_SAMPLE_CHANGE) == -math.inf
    assert compute_checkpoint_selection_score(unmeasurable, MAX_CLEAN_SAMPLE_CHANGE) == 28.0


def make_fake_trainer(checkpoint_directory: Path, tiny_model_kwargs: dict) -> SimpleNamespace:
    final_ema_model = GlareRemovalNAFNet(**tiny_model_kwargs)
    return SimpleNamespace(
        run_directory=SimpleNamespace(checkpoint_directory=checkpoint_directory),
        ema=SimpleNamespace(averaged_model=final_ema_model),
    )


def test_export_uses_best_ema_checkpoint(tmp_path, tiny_model_kwargs):
    trainer = make_fake_trainer(tmp_path, tiny_model_kwargs)
    best_model = GlareRemovalNAFNet(**tiny_model_kwargs)
    with torch.no_grad():
        for parameter in best_model.parameters():
            parameter.fill_(0.123)
    model_config = {"NAME": "GlareRemovalNAFNet", **{key.upper(): value for key, value in tiny_model_kwargs.items()}}
    save_checkpoint_atomically(
        {"step": 7, "model": {}, "ema_model": best_model.state_dict(), "model_config": model_config}, tmp_path / BEST_CHECKPOINT_NAME
    )
    exported_model = select_model_to_export(trainer)
    assert exported_model is not trainer.ema.averaged_model
    for exported_parameter in exported_model.parameters():
        assert torch.all(exported_parameter == 0.123)


def test_export_falls_back_to_final_ema_without_a_best_checkpoint(tmp_path, tiny_model_kwargs):
    trainer = make_fake_trainer(tmp_path, tiny_model_kwargs)
    assert select_model_to_export(trainer) is trainer.ema.averaged_model
