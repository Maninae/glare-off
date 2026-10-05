"""Config composition rules and a tiny CPU end-to-end train -> checkpoint -> resume run."""

from pathlib import Path

import pytest
import torch
from omegaconf.errors import ConfigKeyError

from glare_model.training.checkpointing import LAST_CHECKPOINT_NAME, load_checkpoint
from glare_model.training.config_loading import load_training_config
from glare_model.training.glare_model_trainer import GlareModelTrainer
from glare_model.training.optimization import build_warmup_cosine_scheduler
from glare_model.training.run_directory import compute_run_hash

BASE_CONFIG_PATH = Path("configs/glare_model/base.yaml")


def tiny_cpu_overrides(run_directory: Path, max_iter: int) -> list[str]:
    """Dotlist turning the base config into a seconds-long CPU run on fake 64x128 crops."""
    return [
        "CONFIG_GROUPS.MODEL=small",
        "MODEL.BASE_WIDTH=8",
        "DATA.CROP_HEIGHT=64",
        "DATA.CROP_WIDTH=128",
        "DATA.SOURCE_FACE_PROVIDER.TRAIN.FACE_COUNT=8",
        "DATA.SOURCE_FACE_PROVIDER.VAL.FACE_COUNT=2",
        "DATALOADER.BATCH_SIZE=2",
        "DATALOADER.VAL_BATCH_SIZE=2",
        "DATALOADER.NUM_WORKERS=0",
        "SOLVER.GRADIENT_ACCUMULATION_STEPS=2",
        f"SOLVER.MAX_ITER={max_iter}",
        "SOLVER.WARMUP_ITERS=1",
        "TRAIN.DEVICE=cpu",
        "TRAIN.LOG_PERIOD=1",
        "TRAIN.VAL_PERIOD=2",
        "TRAIN.CHECKPOINT_PERIOD=2",
        f"OUTPUT.RUN_DIRECTORY={run_directory}",
    ]


def test_config_groups_compose_and_cli_selects_the_group():
    base_config = load_training_config(BASE_CONFIG_PATH, [])
    small_config = load_training_config(BASE_CONFIG_PATH, ["CONFIG_GROUPS.MODEL=small"])
    assert list(base_config.MODEL.ENCODER_BLOCK_COUNTS) == [1, 1, 2, 4]
    assert small_config.MODEL.MIDDLE_BLOCK_COUNT == 1
    assert small_config.DATA.GLARE_SYNTHESIZER.NAME == "build_fake_blob_glare_synthesizer"


def test_real_data_group_names_the_real_modules():
    real_config = load_training_config(BASE_CONFIG_PATH, ["CONFIG_GROUPS.DATA=real"])
    assert real_config.DATA.SOURCE_FACE_PROVIDER.TRAIN.NAME == "ManifestSourceFaceProvider"


def test_typo_in_override_fails_loudly():
    with pytest.raises(ConfigKeyError):
        load_training_config(BASE_CONFIG_PATH, ["SOLVER.BASE_LRR=0.1"])


def test_more_than_four_workers_is_rejected():
    with pytest.raises(ValueError, match="NUM_WORKERS"):
        load_training_config(BASE_CONFIG_PATH, ["DATALOADER.NUM_WORKERS=6"])


def test_run_hash_ignores_worker_count_but_not_learning_rate():
    base_hash = compute_run_hash(load_training_config(BASE_CONFIG_PATH, []))
    assert compute_run_hash(load_training_config(BASE_CONFIG_PATH, ["DATALOADER.NUM_WORKERS=1"])) == base_hash
    assert compute_run_hash(load_training_config(BASE_CONFIG_PATH, ["SOLVER.BASE_LR=0.1"])) != base_hash


def test_warmup_cosine_schedule_endpoints():
    solver_config = load_training_config(BASE_CONFIG_PATH, ["SOLVER.WARMUP_ITERS=10", "SOLVER.MAX_ITER=110"]).SOLVER
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=solver_config.BASE_LR)
    scheduler = build_warmup_cosine_scheduler(optimizer, solver_config)
    multiplier = scheduler.lr_lambdas[0]
    assert multiplier(0) == pytest.approx(0.1)
    assert multiplier(10) == pytest.approx(1.0)
    assert multiplier(110) == pytest.approx(solver_config.MIN_LR / solver_config.BASE_LR)


def test_tiny_training_run_checkpoints_and_resumes(tmp_path):
    run_directory = tmp_path / "run"
    first_trainer = GlareModelTrainer(load_training_config(BASE_CONFIG_PATH, tiny_cpu_overrides(run_directory, max_iter=2)))
    first_metrics = first_trainer.train()
    assert first_trainer.step == 2
    assert {"psnr_lens", "ssim_all", "iou_glare_mask"} <= set(first_metrics)
    assert (run_directory / "config.yaml").exists()
    assert load_checkpoint(run_directory / "checkpoints" / LAST_CHECKPOINT_NAME)["step"] == 2
    assert any((run_directory / "tb").iterdir())

    resumed_trainer = GlareModelTrainer(load_training_config(BASE_CONFIG_PATH, tiny_cpu_overrides(run_directory, max_iter=4)))
    assert resumed_trainer.step == 2
    for resumed_parameter, saved_parameter in zip(resumed_trainer.model.parameters(), first_trainer.model.parameters()):
        assert torch.equal(resumed_parameter, saved_parameter)
    resumed_trainer.train()
    assert load_checkpoint(run_directory / "checkpoints" / LAST_CHECKPOINT_NAME)["step"] == 4
