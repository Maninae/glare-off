"""Training-time augmentations around the eye crop: detector-noise jitter, photometric jitter, flip.

All functions are pure and take an explicit `np.random.Generator`, so a fixed seed reproduces a
sample exactly (validation relies on this).

- Eye-center jitter mimics YuNet's error at inference: independent per-eye noise (a few percent
  of eye distance) plus a small extra roll, scale, and shift of the pair.
- Photometric jitter is applied to the CLEAN crop before glare synthesis, so target and input
  share the same exposure and color; it widens the skin-tone / white-balance distribution.
- The horizontal flip mirrors the crop and swaps lens labels 1 <-> 2 so "image-left lens" stays
  true after the flip.
"""

from dataclasses import dataclass

import numpy as np

from glare_model.data.source_face import LENS_LABEL_IMAGE_LEFT, LENS_LABEL_IMAGE_RIGHT


@dataclass
class EyeCenterJitterSettings:
    """Magnitudes of the eye-center jitter; all distances are fractions of the eye distance."""

    per_eye_noise_fraction: float = 0.03
    roll_degrees: float = 3.0
    scale_fraction: float = 0.05
    shift_fraction: float = 0.03


@dataclass
class PhotometricJitterSettings:
    """Half-widths of the uniform photometric jitter ranges, plus the flip probability."""

    brightness: float = 0.15
    contrast: float = 0.15
    gamma: float = 0.15
    saturation: float = 0.2
    white_balance: float = 0.06
    horizontal_flip_probability: float = 0.5


def jitter_eye_centers(
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
    jitter_settings: EyeCenterJitterSettings,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return perturbed (left, right) eye centers, keeping the left eye on the image's left."""
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_distance = float(np.linalg.norm(right_eye - left_eye))

    noise_scale = jitter_settings.per_eye_noise_fraction * eye_distance
    left_eye = left_eye + random_generator.normal(0.0, noise_scale, size=2)
    right_eye = right_eye + random_generator.normal(0.0, noise_scale, size=2)

    # Rotate and scale the pair about its midpoint, then shift it.
    midpoint = (left_eye + right_eye) / 2.0
    roll_radians = np.radians(random_generator.uniform(-jitter_settings.roll_degrees, jitter_settings.roll_degrees))
    scale = 1.0 + random_generator.uniform(-jitter_settings.scale_fraction, jitter_settings.scale_fraction)
    rotation_scale = scale * np.array(
        [[np.cos(roll_radians), -np.sin(roll_radians)], [np.sin(roll_radians), np.cos(roll_radians)]]
    )
    shift = random_generator.uniform(-1.0, 1.0, size=2) * jitter_settings.shift_fraction * eye_distance
    left_eye = midpoint + rotation_scale @ (left_eye - midpoint) + shift
    right_eye = midpoint + rotation_scale @ (right_eye - midpoint) + shift
    return left_eye, right_eye


def apply_photometric_jitter(
    clean_eye_crop: np.ndarray, jitter_settings: PhotometricJitterSettings, random_generator: np.random.Generator
) -> np.ndarray:
    """Return the crop with random brightness, contrast, gamma, saturation, and white balance.

    Input and output are (H, W, 3) float32 sRGB in [0, 1].
    """

    def draw_factor(half_width: float) -> float:
        return 1.0 + random_generator.uniform(-half_width, half_width)

    adjusted = clean_eye_crop * draw_factor(jitter_settings.brightness)
    adjusted = adjusted * np.array([draw_factor(jitter_settings.white_balance) for _ in range(3)], dtype=np.float32)
    mean_level = adjusted.mean()
    adjusted = (adjusted - mean_level) * draw_factor(jitter_settings.contrast) + mean_level
    gray = adjusted.mean(axis=2, keepdims=True)
    adjusted = gray + (adjusted - gray) * draw_factor(jitter_settings.saturation)
    adjusted = np.clip(adjusted, 0.0, 1.0) ** draw_factor(jitter_settings.gamma)
    return adjusted.astype(np.float32)


def flip_crop_and_lens_labels_horizontally(
    clean_eye_crop: np.ndarray, lens_label_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror both arrays left-right and swap the image-left / image-right lens labels."""
    flipped_crop = np.ascontiguousarray(clean_eye_crop[:, ::-1])
    flipped_labels = np.ascontiguousarray(lens_label_mask[:, ::-1]).copy()
    was_left = flipped_labels == LENS_LABEL_IMAGE_LEFT
    was_right = flipped_labels == LENS_LABEL_IMAGE_RIGHT
    flipped_labels[was_left] = LENS_LABEL_IMAGE_RIGHT
    flipped_labels[was_right] = LENS_LABEL_IMAGE_LEFT
    return flipped_crop, flipped_labels
