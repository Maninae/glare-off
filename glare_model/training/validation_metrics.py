"""Validation metrics on the BLENDED output (what the app shows), accumulated over a val pass.

Per batch the accumulator adds sums, so the final numbers are pixel-weighted over the whole
val set (not a mean of per-batch means):
- PSNR / SSIM over the full crop and inside the lens mask.
- Input PSNR inside the lens: the "do nothing" baseline, so a gain is readable at a glance.
- IoU of the glare and lost-detail masks, both thresholded at 0.5.
"""

import math

import torch

from glare_model.losses.structural_similarity import compute_ssim_map

MASK_THRESHOLD = 0.5
PSNR_MSE_FLOOR = 1e-10
MASK_CHANNEL_NAMES = ("glare_mask", "lost_detail_mask")


class ValidationMetricAccumulator:
    """Sums per-pixel errors and mask overlaps across batches; `summarize` turns them into metrics."""

    def __init__(self):
        self.metric_sums: dict[str, float] = {}

    def add_to(self, metric_name: str, value: float) -> None:
        """Add `value` to the running sum `metric_name`."""
        self.metric_sums[metric_name] = self.metric_sums.get(metric_name, 0.0) + float(value)

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

        mask_targets = torch.cat([batch_data_dict["glare_mask"], batch_data_dict["lost_detail_mask"]], dim=1)
        predicted_binary = predicted_masks >= MASK_THRESHOLD
        target_binary = mask_targets >= MASK_THRESHOLD
        for channel_index, channel_name in enumerate(MASK_CHANNEL_NAMES):
            self.add_to(f"{channel_name}_intersection", (predicted_binary[:, channel_index] & target_binary[:, channel_index]).sum())
            self.add_to(f"{channel_name}_union", (predicted_binary[:, channel_index] | target_binary[:, channel_index]).sum())

    def summarize(self) -> dict[str, float]:
        """Return {metric_name: value}; IoU is NaN when a mask never appears (union 0)."""
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
        }
        for channel_name in MASK_CHANNEL_NAMES:
            union = sums[f"{channel_name}_union"]
            metrics[f"iou_{channel_name}"] = sums[f"{channel_name}_intersection"] / union if union > 0 else float("nan")
        return metrics
