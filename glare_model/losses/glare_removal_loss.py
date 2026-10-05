"""GlareRemovalLoss: every training term, computed on the image the USER will see.

The app blends `result = glare + glare_mask * (clean - glare)` (Model I/O contract). Since the
network's clean output is `glare + delta`, the blended result is `glare + predicted_glare_mask *
delta`; the reconstruction terms score that blend, so the restoration head and the mask head are
trained jointly for the final picture rather than separately for intermediate outputs.

Terms (weights in `LOSS.WEIGHTS`, zero disables a term):
- BLENDED_L1: L1 of blended vs clean over the whole crop.
- LENS_L1 / GLARE_L1: the same error, averaged only inside the lens mask / ground-truth glare mask,
  so small in-lens errors are not diluted by the big untouched face.
- UNBLENDED_LENS_L1: L1 of the raw clean output inside the lens. Keeps `delta` learning where the
  predicted mask is still near zero early in training (otherwise its gradient is gated to 0).
- SSIM: 1 - SSIM of blended vs clean.
- FFT: L1 between the orthonormal 2D FFT amplitudes of blended and clean (penalizes blur and
  leftover low-frequency veil).
- MASK_FOCAL / MASK_DICE: focal BCE (alpha 0.25, gamma 2, soft targets) and soft Dice on both
  mask channels.
- OUTSIDE_LENS_CHANGE: mean |delta| outside the lens. Pixels outside the lens must not change at
  all; the target there may differ from the input (JPEG, noise), so this compares to the INPUT.
- PERCEPTUAL_VGG: disabled hook; a VGG/LPIPS loss is a licensing gray area (docs/research/03).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from glare_model.losses.structural_similarity import compute_ssim_map

MASKED_MEAN_EPSILON = 1.0
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_SMOOTHING = 1.0


@dataclass
class GlareRemovalLossWeights:
    """Per-term weights; field names mirror the UPPERCASE `LOSS.WEIGHTS` config keys."""

    blended_l1: float = 1.0
    lens_l1: float = 2.0
    glare_l1: float = 2.0
    unblended_lens_l1: float = 0.5
    ssim: float = 0.2
    fft: float = 0.05
    mask_focal: float = 1.0
    mask_dice: float = 0.5
    outside_lens_change: float = 1.0
    perceptual_vgg: float = 0.0


def masked_mean(values: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` (N, C, H, W) over pixels weighted by `weight_mask` (N, 1, H, W).

    The +1 in the denominator keeps samples with an empty mask (no glasses, no glare) at ~0 loss
    instead of dividing by zero.
    """
    weighted_sum = (values * weight_mask).sum()
    weight_total = weight_mask.sum() * values.shape[1]
    return weighted_sum / (weight_total + MASKED_MEAN_EPSILON)


def soft_target_focal_loss(mask_logits: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
    """Focal BCE (Lin et al. 2017) generalized to soft targets in [0, 1], averaged over pixels."""
    binary_cross_entropy = functional.binary_cross_entropy_with_logits(mask_logits, mask_targets, reduction="none")
    predicted_probability = torch.sigmoid(mask_logits)
    modulating_factor = (mask_targets - predicted_probability).abs() ** FOCAL_GAMMA
    alpha_weight = FOCAL_ALPHA * mask_targets + (1 - FOCAL_ALPHA) * (1 - mask_targets)
    return (alpha_weight * modulating_factor * binary_cross_entropy).mean()


def soft_dice_loss(mask_probabilities: torch.Tensor, mask_targets: torch.Tensor) -> torch.Tensor:
    """1 - soft Dice per channel over the whole batch, averaged over channels with any positive pixel.

    - Batch-level, not per sample: glare-free samples have empty targets, and per-sample Dice
      would punish their tiny background probabilities as hard as a missed glare spot.
    - A channel with no positive pixel in the batch contributes nothing (focal loss covers it).
    """
    intersection = (mask_probabilities * mask_targets).sum(dim=(0, 2, 3))
    target_total = mask_targets.sum(dim=(0, 2, 3))
    total = mask_probabilities.sum(dim=(0, 2, 3)) + target_total
    dice_loss_per_channel = 1 - (2 * intersection + DICE_SMOOTHING) / (total + DICE_SMOOTHING)
    has_positive_pixels = (target_total >= 1.0).to(dice_loss_per_channel.dtype)
    return (dice_loss_per_channel * has_positive_pixels).sum() / has_positive_pixels.sum().clamp(min=1.0)


def fft_amplitude_l1(predicted_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    """L1 between orthonormal 2D FFT amplitudes."""
    predicted_amplitude = torch.fft.rfft2(predicted_image, norm="ortho").abs()
    target_amplitude = torch.fft.rfft2(target_image, norm="ortho").abs()
    return (predicted_amplitude - target_amplitude).abs().mean()


def blend_prediction_like_the_app(
    glare_eye_crop: torch.Tensor, restoration_delta: torch.Tensor, predicted_glare_mask: torch.Tensor
) -> torch.Tensor:
    """Return `glare + glare_mask * (clean - glare)` with `clean = glare + delta` (unclipped)."""
    return glare_eye_crop + predicted_glare_mask * restoration_delta


class GlareRemovalLoss:
    """Callable computing every term and the weighted total from raw network outputs and a batch."""

    def __init__(self, loss_weights: GlareRemovalLossWeights):
        if loss_weights.perceptual_vgg > 0:
            raise NotImplementedError(
                "PERCEPTUAL_VGG is a disabled hook: VGG/LPIPS weights are a licensing gray area "
                "(docs/research/03-training-data.md, Q5). Implement deliberately before enabling."
            )
        self.loss_weights = loss_weights

    def __call__(
        self, restoration_delta: torch.Tensor, mask_logits: torch.Tensor, batch_data_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return {"loss_<term>": unweighted value, ..., "loss_total": weighted sum}."""
        glare_eye_crop = batch_data_dict["glare_eye_crop"]
        clean_eye_crop = batch_data_dict["clean_eye_crop"]
        lens_mask = batch_data_dict["lens_mask"]
        glare_mask = batch_data_dict["glare_mask"]
        mask_targets = torch.cat([glare_mask, batch_data_dict["lost_detail_mask"]], dim=1)

        mask_probabilities = torch.sigmoid(mask_logits)
        blended_crop = blend_prediction_like_the_app(glare_eye_crop, restoration_delta, mask_probabilities[:, :1])
        blended_error = (blended_crop - clean_eye_crop).abs()
        unblended_error = (glare_eye_crop + restoration_delta - clean_eye_crop).abs()

        loss_terms = {
            "blended_l1": blended_error.mean(),
            "lens_l1": masked_mean(blended_error, lens_mask),
            "glare_l1": masked_mean(blended_error, glare_mask),
            "unblended_lens_l1": masked_mean(unblended_error, lens_mask),
            "ssim": 1 - compute_ssim_map(blended_crop, clean_eye_crop).mean(),
            "fft": fft_amplitude_l1(blended_crop, clean_eye_crop),
            "mask_focal": soft_target_focal_loss(mask_logits, mask_targets),
            "mask_dice": soft_dice_loss(mask_probabilities, mask_targets),
            "outside_lens_change": masked_mean(restoration_delta.abs(), 1 - lens_mask),
        }
        weighted_total = sum(getattr(self.loss_weights, name) * value for name, value in loss_terms.items())
        loss_outputs = {f"loss_{name}": value for name, value in loss_terms.items()}
        loss_outputs["loss_total"] = weighted_total
        return loss_outputs
