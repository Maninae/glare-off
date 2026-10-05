"""Ground-truth masks: where glare changed the pixel, and where it destroyed the detail underneath.

Both masks are computed from the NOISE-FREE, JPEG-FREE render, so they describe the glare itself
rather than the random grain added on top of it.

- `glare_mask`: smoothstep of the largest per-channel sRGB change, from GLARE_MASK_CHANGE_LOW
  (about 3 code values, invisible) to GLARE_MASK_CHANGE_HIGH (about 15, clearly visible).
- `lost_detail_mask`: remaining contrast of the transmitted signal, i.e. how much a small change in
  the clean pixel still moves the glared pixel, d(glared sRGB) / d(clean sRGB), by the chain rule:
      decode slope at the clean value  x  d(glared linear)/d(clean linear)  x  encode slope at the glared value
  with d(glared linear)/d(clean linear) = 1 + T'(clean*(1-a) + R)*(1-a) - T'(clean) (T = highlight
  response, a = attenuation, R = reflection), and 0 where the output is clipped (within half an
  8-bit code value of white, which an 8-bit photo stores as 255). The largest value over
  channels counts (one surviving channel still carries detail). Clipping gives 0; a veil R over a dark
  iris gives about (L_clean / L_glared)^(1/2.4). Contrast at or below LOST_DETAIL_CONTRAST_FULL
  (detail shrunk 10x, under the noise and 8-bit floor) is fully lost; the mask ramps to 0 at
  LOST_DETAIL_CONTRAST_NONE (4x). It is then capped by `glare_mask`, so it is always a subset.
"""

import numpy as np

from glare_synthesis.camera_response import highlight_response_slope
from glare_synthesis.color_science import srgb_decoding_slope, srgb_encoding_slope, srgb_to_linear

GLARE_MASK_CHANGE_LOW = 0.012
GLARE_MASK_CHANGE_HIGH = 0.06
LOST_DETAIL_CONTRAST_FULL = 0.1
LOST_DETAIL_CONTRAST_NONE = 0.25
# An 8-bit photo stores anything within half a code value of white as 255, i.e. clipped.
CLIPPED_SRGB_LEVEL = 1.0 - 0.5 / 255.0
CLIPPED_LINEAR_LEVEL = float(srgb_to_linear(np.array([CLIPPED_SRGB_LEVEL]))[0])


def smoothstep(edge_low: float, edge_high: float, values: np.ndarray) -> np.ndarray:
    """Cubic Hermite ramp from 0 at edge_low to 1 at edge_high (works for edge_low > edge_high too)."""
    ramp = np.clip((values - edge_low) / (edge_high - edge_low), 0.0, 1.0)
    return ramp * ramp * (3.0 - 2.0 * ramp)


def compute_glare_mask(clean_srgb: np.ndarray, glared_srgb_noise_free: np.ndarray) -> np.ndarray:
    """Soft (H, W) mask, 1 where the glare visibly changed the pixel."""
    largest_channel_change = np.abs(glared_srgb_noise_free - clean_srgb).max(axis=-1)
    return smoothstep(GLARE_MASK_CHANGE_LOW, GLARE_MASK_CHANGE_HIGH, largest_channel_change).astype(np.float32)


def compute_remaining_detail_contrast(clean_srgb: np.ndarray, clean_linear: np.ndarray, glared_linear_noise_free: np.ndarray,
                                      reflection_linear: np.ndarray, attenuation: np.ndarray, soft_knee_start: float | None) -> np.ndarray:
    """d(glared sRGB) / d(clean sRGB) per pixel, max over channels: 1 = untouched detail, 0 = clipped flat."""
    exposed_linear = clean_linear * (1.0 - attenuation) + reflection_linear
    linear_slope = 1.0 + highlight_response_slope(exposed_linear, soft_knee_start) * (1.0 - attenuation) - highlight_response_slope(clean_linear, soft_knee_start)
    linear_slope = np.where(glared_linear_noise_free >= CLIPPED_LINEAR_LEVEL, 0.0, np.maximum(linear_slope, 0.0))
    srgb_slope = srgb_decoding_slope(clean_srgb) * linear_slope * srgb_encoding_slope(glared_linear_noise_free)
    return srgb_slope.max(axis=-1)


def compute_lost_detail_mask(remaining_detail_contrast: np.ndarray, glare_mask: np.ndarray) -> np.ndarray:
    """Soft (H, W) mask, 1 where the transmitted detail is unrecoverable; never exceeds `glare_mask`."""
    lost_detail = smoothstep(LOST_DETAIL_CONTRAST_NONE, LOST_DETAIL_CONTRAST_FULL, remaining_detail_contrast)
    return np.minimum(lost_detail, glare_mask).astype(np.float32)
