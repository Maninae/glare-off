"""Validation metrics on the BLENDED output (what the app shows), accumulated over a val pass.

Per batch the accumulator adds sums, so the final numbers are pixel-weighted over the whole
val set (not a mean of per-batch means):
- PSNR / SSIM over the full crop and inside the lens mask.
- Input PSNR inside the lens: the "do nothing" baseline, so a gain is readable at a glance.
- IoU of the glare and lost-detail masks, both thresholded at 0.5.
- Change guards (|blended - input|, mean over values): `clean_sample_change` (+ `_max`) over
  glare-free samples (empty target glare mask), and `outside_lens_change` over all samples,
  weighted `(1 - lens_mask) * (1 - glare_mask)` like the loss term. PSNR alone is blind here:
  it averages a few spurious edits away, and the lens PSNR never sees outside the lens.

Best-checkpoint rule (`compute_checkpoint_selection_score`): the score is `psnr_lens`, but a
checkpoint whose `clean_sample_change` exceeds `TRAIN.BEST_MAX_CLEAN_SAMPLE_CHANGE` scores -inf
and is never kept as best. If the val set has no glare-free sample the gate cannot be measured
and only `psnr_lens` decides.
"""

import math

import torch

from glare_model.losses.structural_similarity import compute_ssim_map

MASK_THRESHOLD = 0.5
PSNR_MSE_FLOOR = 1e-10
MASK_CHANNEL_NAMES = ("glare_mask", "lost_detail_mask")


def compute_checkpoint_selection_score(metrics: dict[str, float], max_clean_sample_change: float) -> float:
    """Return `psnr_lens`, or -inf when the checkpoint edits glare-free samples more than allowed."""
    clean_sample_change = metrics.get("clean_sample_change", float("nan"))
    if not math.isnan(clean_sample_change) and clean_sample_change > max_clean_sample_change:
        return -math.inf
    return metrics["psnr_lens"]


class ValidationMetricAccumulator:
    """Sums per-pixel errors and mask overlaps across batches; `summarize` turns them into metrics."""

    def __init__(self):
        self.metric_sums: dict[str, float] = {}
        self.metric_maxima: dict[str, float] = {}

    def add_to(self, metric_name: str, value: float) -> None:
        """Add `value` to the running sum `metric_name`."""
        self.metric_sums[metric_name] = self.metric_sums.get(metric_name, 0.0) + float(value)

    def raise_to(self, metric_name: str, value: float) -> None:
        """Keep the running maximum of `metric_name`."""
        self.metric_maxima[metric_name] = max(self.metric_maxima.get(metric_name, -math.inf), float(value))

    @torch.no_grad()
    def update(self, blended_crop: torch.Tensor, predicted_masks: torch.Tensor, batch_data_dict: dict[str, torch.Tensor]) -> None:
        """Accumulate one batch. `blended_crop` is clipped to [0, 1]; masks are probabilities."""
        clean_eye_crop = batch_data_dict["clean_eye_crop"]
        lens_mask = batch_data_dict["lens_mask"]
        channel_count = clean_eye_crop.shape[1]
        squared_error = ((blended_crop - clean_eye_crop) ** 2).sum(dim=1, keepdim=True)
        input_squared_error = ((batch_data_dict["glare_eye_crop"] - clean_eye_crop) ** 2).sum(dim=1, keepdim=True)
        ssim_map = compute_ssim_map(blended_crop, clean_eye_crop)

        self.add_to("squared_error_all", squared_error.sum())
        self.add_to("value_count_all", squared_error.numel() * channel_count)
        self.add_to("squared_error_lens", (squared_error * lens_mask).sum())
        self.add_to("input_squared_error_lens", (input_squared_error * lens_mask).sum())
        self.add_to("value_count_lens", lens_mask.sum() * channel_count)
        self.add_to("ssim_sum_all", ssim_map.sum())
        self.add_to("pixel_count_all", ssim_map.numel())
        self.add_to("ssim_sum_lens", (ssim_map * lens_mask).sum())
        self.add_to("pixel_count_lens", lens_mask.sum())

        glare_mask = batch_data_dict["glare_mask"]
        absolute_change = (blended_crop - batch_data_dict["glare_eye_crop"]).abs()
        outside_lens_weight = (1 - lens_mask) * (1 - glare_mask)
        self.add_to("outside_lens_change_sum", (absolute_change * outside_lens_weight).sum())
        self.add_to("outside_lens_value_count", outside_lens_weight.sum() * channel_count)
        is_glare_free = glare_mask.flatten(1).amax(dim=1) == 0
        if is_glare_free.any():
            clean_sample_change = absolute_change[is_glare_free]
            self.add_to("clean_sample_change_sum", clean_sample_change.sum())
            self.add_to("clean_sample_value_count", clean_sample_change.numel())
            self.raise_to("clean_sample_change_max", clean_sample_change.max())

        mask_targets = torch.cat([glare_mask, batch_data_dict["lost_detail_mask"]], dim=1)
        predicted_binary = predicted_masks >= MASK_THRESHOLD
        target_binary = mask_targets >= MASK_THRESHOLD
        for channel_index, channel_name in enumerate(MASK_CHANNEL_NAMES):
            self.add_to(f"{channel_name}_intersection", (predicted_binary[:, channel_index] & target_binary[:, channel_index]).sum())
            self.add_to(f"{channel_name}_union", (predicted_binary[:, channel_index] | target_binary[:, channel_index]).sum())

    def summarize(self) -> dict[str, float]:
        """Return {metric_name: value}; IoU is NaN when a mask never appears (union 0), clean-sample change when no val sample is glare-free."""
        sums = self.metric_sums

        def psnr(squared_error_key: str, count_key: str) -> float:
            mean_squared_error = sums[squared_error_key] / max(sums[count_key], 1.0)
            return 10.0 * math.log10(1.0 / max(mean_squared_error, PSNR_MSE_FLOOR))

        metrics = {
            "psnr_all": psnr("squared_error_all", "value_count_all"),
            "psnr_lens": psnr("squared_error_lens", "value_count_lens"),
            "input_psnr_lens": psnr("input_squared_error_lens", "value_count_lens"),
            "ssim_all": sums["ssim_sum_all"] / max(sums["pixel_count_all"], 1.0),
            "ssim_lens": sums["ssim_sum_lens"] / max(sums["pixel_count_lens"], 1.0),
            "outside_lens_change": sums["outside_lens_change_sum"] / max(sums["outside_lens_value_count"], 1.0),
        }
        clean_value_count = sums.get("clean_sample_value_count", 0.0)
        metrics["clean_sample_change"] = sums["clean_sample_change_sum"] / clean_value_count if clean_value_count else float("nan")
        metrics["clean_sample_change_max"] = self.metric_maxima.get("clean_sample_change_max", float("nan"))
        for channel_name in MASK_CHANNEL_NAMES:
            union = sums[f"{channel_name}_union"]
            metrics[f"iou_{channel_name}"] = sums[f"{channel_name}_intersection"] / union if union > 0 else float("nan")
        return metrics
