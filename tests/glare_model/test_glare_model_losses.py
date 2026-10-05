"""Loss sanity: zero at the target, each guard term fires when it should, the VGG hook stays off."""

import pytest
import torch

from glare_model.losses.glare_removal_loss import GlareRemovalLoss, GlareRemovalLossWeights, masked_mean
from glare_model.losses.structural_similarity import compute_ssim_map

PERFECT_LOGIT_MAGNITUDE = 40.0


def make_batch_with_glare_in_lens() -> dict[str, torch.Tensor]:
    """A 2-sample batch: binary lens and glare masks, glare added only inside the lens."""
    torch.manual_seed(0)
    clean_eye_crop = torch.rand(2, 3, 32, 64) * 0.5
    lens_mask = torch.zeros(2, 1, 32, 64)
    lens_mask[..., 8:24, 8:56] = 1.0
    glare_mask = torch.zeros_like(lens_mask)
    glare_mask[..., 10:20, 10:30] = 1.0
    lost_detail_mask = torch.zeros_like(lens_mask)
    lost_detail_mask[..., 12:16, 12:20] = 1.0
    glare_eye_crop = clean_eye_crop + 0.4 * glare_mask
    return {
        "clean_eye_crop": clean_eye_crop,
        "glare_eye_crop": glare_eye_crop,
        "glare_mask": glare_mask,
        "lost_detail_mask": lost_detail_mask,
        "lens_mask": lens_mask,
    }


def perfect_logits(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Logits whose sigmoid is (numerically) exactly the binary mask targets."""
    targets = torch.cat([batch["glare_mask"], batch["lost_detail_mask"]], dim=1)
    return (targets * 2 - 1) * PERFECT_LOGIT_MAGNITUDE


def test_every_term_is_zero_when_prediction_equals_target():
    batch = make_batch_with_glare_in_lens()
    loss_outputs = GlareRemovalLoss(GlareRemovalLossWeights())(
        batch["clean_eye_crop"] - batch["glare_eye_crop"], perfect_logits(batch), batch
    )
    for term_name, term_value in loss_outputs.items():
        assert float(term_value) == pytest.approx(0.0, abs=1e-5), term_name


def test_outside_lens_change_fires_on_any_change_outside_the_lens():
    batch = make_batch_with_glare_in_lens()
    restoration_delta = batch["clean_eye_crop"] - batch["glare_eye_crop"]
    restoration_delta[..., 0:4, 0:4] += 0.1  # outside the lens
    loss_outputs = GlareRemovalLoss(GlareRemovalLossWeights())(restoration_delta, perfect_logits(batch), batch)
    assert float(loss_outputs["loss_outside_lens_change"]) > 0.0


def test_doing_nothing_is_penalized_inside_the_glare():
    batch = make_batch_with_glare_in_lens()
    no_change = torch.zeros_like(batch["glare_eye_crop"])
    loss_outputs = GlareRemovalLoss(GlareRemovalLossWeights())(no_change, perfect_logits(batch), batch)
    assert float(loss_outputs["loss_glare_l1"]) > 0.1
    assert float(loss_outputs["loss_outside_lens_change"]) == 0.0


def test_perceptual_vgg_hook_is_disabled():
    with pytest.raises(NotImplementedError, match="PERCEPTUAL_VGG"):
        GlareRemovalLoss(GlareRemovalLossWeights(perceptual_vgg=0.01))


def test_ssim_of_identical_images_is_one():
    image = torch.rand(1, 3, 32, 32)
    torch.testing.assert_close(compute_ssim_map(image, image), torch.ones(1, 1, 32, 32), atol=1e-5, rtol=0)


def test_masked_mean_of_empty_mask_is_zero_not_nan():
    assert float(masked_mean(torch.ones(1, 3, 4, 4), torch.zeros(1, 1, 4, 4))) == 0.0
