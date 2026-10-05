"""Differentiable SSIM (Wang et al. 2004) with an 11x11 Gaussian window, shared by loss and metrics.

Returns the per-pixel SSIM map so callers can average it over any mask (whole crop, inside the
lens). Images are expected in [0, 1]; the stability constants assume a data range of 1.
"""

import torch
import torch.nn.functional as functional

SSIM_WINDOW_SIZE = 11
SSIM_GAUSSIAN_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def build_gaussian_window(channel_count: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a (C, 1, 11, 11) normalized Gaussian kernel for a depthwise conv."""
    offsets = torch.arange(SSIM_WINDOW_SIZE, device=device, dtype=dtype) - SSIM_WINDOW_SIZE // 2
    profile = torch.exp(-(offsets**2) / (2 * SSIM_GAUSSIAN_SIGMA**2))
    profile = profile / profile.sum()
    window = profile[:, None] * profile[None, :]
    return window.expand(channel_count, 1, SSIM_WINDOW_SIZE, SSIM_WINDOW_SIZE).contiguous()


def compute_ssim_map(predicted_image: torch.Tensor, target_image: torch.Tensor) -> torch.Tensor:
    """Return the (N, 1, H, W) SSIM map, averaged over channels, with reflect padding (same size as input)."""
    channel_count = predicted_image.shape[1]
    window = build_gaussian_window(channel_count, predicted_image.device, predicted_image.dtype)
    padding = SSIM_WINDOW_SIZE // 2

    def local_mean(image: torch.Tensor) -> torch.Tensor:
        padded = functional.pad(image, (padding, padding, padding, padding), mode="reflect")
        return functional.conv2d(padded, window, groups=channel_count)

    mean_predicted = local_mean(predicted_image)
    mean_target = local_mean(target_image)
    variance_predicted = local_mean(predicted_image * predicted_image) - mean_predicted**2
    variance_target = local_mean(target_image * target_image) - mean_target**2
    covariance = local_mean(predicted_image * target_image) - mean_predicted * mean_target

    stability_c1 = SSIM_K1**2
    stability_c2 = SSIM_K2**2
    ssim_map = ((2 * mean_predicted * mean_target + stability_c1) * (2 * covariance + stability_c2)) / (
        (mean_predicted**2 + mean_target**2 + stability_c1) * (variance_predicted + variance_target + stability_c2)
    )
    return ssim_map.mean(dim=1, keepdim=True)
