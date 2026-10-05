"""One validation pass: metrics on the blended output plus a TensorBoard image grid.

Grid layout, one row per sample: input | blended prediction | target | predicted glare mask |
target glare mask | predicted lost-detail mask.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from glare_model.losses.glare_removal_loss import blend_prediction_like_the_app
from glare_model.training.validation_metrics import ValidationMetricAccumulator

IMAGE_GRID_COLUMN_COUNT = 6


def move_batch_to_device(batch_data_dict: dict, device: torch.device) -> dict:
    """Move every tensor in the batch to `device` (strings such as source_id stay as lists)."""
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch_data_dict.items()}


@torch.no_grad()
def run_validation_pass(
    model: nn.Module, validation_loader: DataLoader, device: torch.device, image_grid_count: int, max_batches: int = 0
) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate `model` on the val loader; return (metrics, image grid of the first batch's samples)."""
    model.eval()
    metric_accumulator = ValidationMetricAccumulator()
    grid_rows = []
    for batch_index, batch_data_dict in enumerate(validation_loader):
        if max_batches and batch_index >= max_batches:
            break
        batch_data_dict = move_batch_to_device(batch_data_dict, device)
        glare_eye_crop = batch_data_dict["glare_eye_crop"]
        restoration_delta, mask_logits = model.predict_delta_and_mask_logits(glare_eye_crop)
        predicted_masks = torch.sigmoid(mask_logits)
        blended_crop = blend_prediction_like_the_app(glare_eye_crop, restoration_delta, predicted_masks[:, :1]).clamp(0.0, 1.0)
        metric_accumulator.update(blended_crop, predicted_masks, batch_data_dict)
        # One sample from each of the first batches, so the grid spans several source faces.
        if len(grid_rows) < image_grid_count:
            grid_rows.append(build_validation_grid_row(blended_crop, predicted_masks, batch_data_dict))
    image_grid = make_grid(torch.cat(grid_rows), nrow=IMAGE_GRID_COLUMN_COUNT, padding=2, pad_value=1.0) if grid_rows else None
    return metric_accumulator.summarize(), image_grid


def build_validation_grid_row(blended_crop: torch.Tensor, predicted_masks: torch.Tensor, batch_data_dict: dict) -> torch.Tensor:
    """Return the `IMAGE_GRID_COLUMN_COUNT` (1-row) panels for the batch's first sample, on the CPU."""

    def as_rgb(single_channel: torch.Tensor) -> torch.Tensor:
        return single_channel.expand(-1, 3, -1, -1)

    first = slice(0, 1)
    panels = [
        batch_data_dict["glare_eye_crop"][first],
        blended_crop[first],
        batch_data_dict["clean_eye_crop"][first],
        as_rgb(predicted_masks[first, :1]),
        as_rgb(batch_data_dict["glare_mask"][first]),
        as_rgb(predicted_masks[first, 1:]),
    ]
    return torch.cat(panels).float().cpu()
