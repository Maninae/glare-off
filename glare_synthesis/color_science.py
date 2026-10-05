"""Colour helpers for glare compositing: sRGB transfer curve, light colour, coating tints.

- All glare light is added in LINEAR RGB; these functions are the only place the sRGB curve
  lives, so the encode/decode pair is exactly inverse-consistent for the renderer.
- Light colours and tints are normalized to unit Rec.709 luminance, so "peak brightness" is set
  independently of hue.
"""

import numpy as np

SRGB_LINEAR_SEGMENT_THRESHOLD = 0.04045
SRGB_LINEAR_SEGMENT_SLOPE = 12.92
SRGB_GAMMA = 2.4
SRGB_OFFSET = 0.055
LINEAR_SEGMENT_THRESHOLD = SRGB_LINEAR_SEGMENT_THRESHOLD / SRGB_LINEAR_SEGMENT_SLOPE

REC709_LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Anti-reflective residual colours seen on real lenses (green is the most common, then blue,
# purple; magenta appears on some premium coatings). RGB, before luminance normalization.
AR_COATING_BASE_TINTS_RGB = {
    "pale_blue": (0.75, 0.85, 1.0),
    "near_white": (0.95, 0.97, 1.0),
    "green": (0.35, 1.0, 0.45),
    "blue": (0.40, 0.60, 1.0),
    "purple": (0.80, 0.45, 1.0),
    "magenta": (1.0, 0.45, 0.85),
}
# Most visible glare reads near-white or pale blue; saturated hues show on faint residuals
# (the renderer additionally whitens the tint where the reflection is bright).
AR_COATING_TINT_PROBABILITIES = {
    "pale_blue": 0.5,
    "near_white": 0.3,
    "green": 0.1,
    "blue": 0.05,
    "purple": 0.04,
    "magenta": 0.01,
}
# AR residual colour is desaturated by broadband reflection off dust, smudges and the other
# surface; mix this fraction of white back in so tints are not neon.
AR_TINT_WHITE_MIX_RANGE = (0.45, 0.85)
TINT_CHANNEL_JITTER = 0.15


def srgb_to_linear(srgb_values: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values in [0, 1] to linear light (IEC 61966-2-1)."""
    srgb_values = np.asarray(srgb_values, dtype=np.float32)
    power_branch = np.maximum(srgb_values, np.float32(0.0)) + np.float32(SRGB_OFFSET)
    power_branch *= np.float32(1.0 / (1.0 + SRGB_OFFSET))
    np.power(power_branch, np.float32(SRGB_GAMMA), out=power_branch)
    is_linear_segment = srgb_values <= SRGB_LINEAR_SEGMENT_THRESHOLD
    power_branch[is_linear_segment] = srgb_values[is_linear_segment] * np.float32(1.0 / SRGB_LINEAR_SEGMENT_SLOPE)
    return power_branch


def linear_to_srgb(linear_values: np.ndarray) -> np.ndarray:
    """Encode linear light in [0, 1] to sRGB values (inputs are clipped to [0, 1] first)."""
    linear_values = np.clip(np.asarray(linear_values, dtype=np.float32), np.float32(0.0), np.float32(1.0))
    power_branch = np.power(linear_values, np.float32(1.0 / SRGB_GAMMA))
    power_branch *= np.float32(1.0 + SRGB_OFFSET)
    power_branch -= np.float32(SRGB_OFFSET)
    is_linear_segment = linear_values <= LINEAR_SEGMENT_THRESHOLD
    power_branch[is_linear_segment] = linear_values[is_linear_segment] * np.float32(SRGB_LINEAR_SEGMENT_SLOPE)
    return power_branch


def srgb_decoding_slope(srgb_values: np.ndarray) -> np.ndarray:
    """d(linear) / d(sRGB) at the given sRGB values."""
    srgb_values = np.asarray(srgb_values, dtype=np.float32)
    power_slope = np.float32(SRGB_GAMMA / (1.0 + SRGB_OFFSET)) * np.power((np.maximum(srgb_values, 0.0) + np.float32(SRGB_OFFSET)) / np.float32(1.0 + SRGB_OFFSET), np.float32(SRGB_GAMMA - 1.0))
    return np.where(srgb_values <= SRGB_LINEAR_SEGMENT_THRESHOLD, np.float32(1.0 / SRGB_LINEAR_SEGMENT_SLOPE), power_slope)


def srgb_encoding_slope(linear_values: np.ndarray) -> np.ndarray:
    """d(sRGB) / d(linear) at the given linear values (clipped to [0, 1])."""
    linear_values = np.clip(np.asarray(linear_values, dtype=np.float32), np.float32(LINEAR_SEGMENT_THRESHOLD), np.float32(1.0))
    power_slope = np.float32((1.0 + SRGB_OFFSET) / SRGB_GAMMA) * np.power(linear_values, np.float32(1.0 / SRGB_GAMMA - 1.0))
    return np.where(linear_values <= LINEAR_SEGMENT_THRESHOLD, np.float32(SRGB_LINEAR_SEGMENT_SLOPE), power_slope)


def normalize_rgb_to_unit_luminance(rgb: np.ndarray) -> np.ndarray:
    """Scale an RGB triple so its Rec.709 luminance is 1."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb / max(float(rgb @ REC709_LUMINANCE_WEIGHTS), 1e-6)


def color_temperature_to_linear_rgb(color_temperature_kelvin: float) -> np.ndarray:
    """Approximate linear-RGB colour of a blackbody at the given CCT, unit luminance.

    Uses Tanner Helland's fit to the CIE blackbody locus in sRGB (good to a few percent between
    1000 K and 40000 K), decoded to linear. The camera white point is assumed to be about 5500 K,
    so 2700 K reads warm orange and 9000 K reads cool blue, as in photos.
    """
    temperature_hundreds = float(np.clip(color_temperature_kelvin, 1000.0, 40000.0)) / 100.0
    if temperature_hundreds <= 66.0:
        red = 255.0
        green = 99.4708025861 * np.log(temperature_hundreds) - 161.1195681661
        blue = 0.0 if temperature_hundreds <= 19.0 else 138.5177312231 * np.log(temperature_hundreds - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * (temperature_hundreds - 60.0) ** -0.1332047592
        green = 288.1221695283 * (temperature_hundreds - 60.0) ** -0.0755148492
        blue = 255.0
    srgb_triple = np.clip(np.array([red, green, blue], dtype=np.float32) / 255.0, 0.0, 1.0)
    # Re-balance to an assumed ~5500 K camera white point instead of the 6500 K display white.
    camera_white = np.array([1.0, 0.93, 0.85], dtype=np.float32)
    linear_triple = srgb_to_linear(srgb_triple) / srgb_to_linear(camera_white)
    return normalize_rgb_to_unit_luminance(linear_triple)


def sample_ar_coating_tint(random_generator: np.random.Generator) -> np.ndarray:
    """Draw an anti-reflective residual tint (linear RGB, unit luminance), jittered and desaturated."""
    tint_names = list(AR_COATING_TINT_PROBABILITIES)
    tint_weights = np.array([AR_COATING_TINT_PROBABILITIES[name] for name in tint_names])
    tint_name = tint_names[random_generator.choice(len(tint_names), p=tint_weights / tint_weights.sum())]
    base_tint = np.array(AR_COATING_BASE_TINTS_RGB[tint_name], dtype=np.float32)
    jittered_tint = base_tint * random_generator.uniform(1.0 - TINT_CHANNEL_JITTER, 1.0 + TINT_CHANNEL_JITTER, size=3)
    white_mix = random_generator.uniform(*AR_TINT_WHITE_MIX_RANGE)
    mixed_tint = (1.0 - white_mix) * normalize_rgb_to_unit_luminance(jittered_tint) + white_mix * np.ones(3, np.float32)
    return normalize_rgb_to_unit_luminance(mixed_tint)
