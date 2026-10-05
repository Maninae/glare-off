"""Simulated phone capture: cut a training crop through the same low-scale resampling path as a phone photo.

Crop scale is crop pixels per photo pixel. Source-manifest faces are small in pixels (eye distance
~150-310 px), so their crops sit at scale ~0.6-1.2. Phone photos have far MORE pixels per eye and
land at ~0.15-0.4, where the app area-pre-shrinks the photo (factor 2-6) before its bilinear warp.

To put a source face on that path, this module enlarges the photo REGION under the crop (never
the whole photo: a 4-8x enlargement of a full photo would cost hundreds of MB) by
`native_scale / target_scale`, so the crop is then cut at the target scale by the same resampler
the app uses (`eye_crop.eye_crop_geometry.extract_eye_crop_with_area_prefilter`), INTER_AREA
pre-shrink included. Lens labels (as one-hot planes) go through the identical region and resize.

- No real detail is created: the enlargement is smooth (cubic). What training gains is the app's
  exact low-scale resampling chain, and noise/JPEG statistics at photo resolution that the
  pre-shrink then averages down, as it does on a real phone photo.
- Degradation (sensor noise, then JPEG re-encode, the camera's order) is applied to the photo
  region BEFORE glare synthesis, so the clean target and the glared input share it.
- The region is handled as uint8, like a decoded phone photo (and like the app's crop input). A
  float32 region at scale 0.15 is ~100 MB per copy; uint8 keeps the whole path near 100 MB.
- All functions are pure and take an explicit `np.random.Generator`.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, eye_crop_scale

UINT8_MAX = 255.0
# Photo pixels kept around the crop footprint: covers the bilinear taps and the area pre-shrink blocks.
REGION_MARGIN_PHOTO_PIXELS = 4
# Rows of the enlarged region given noise per pass, so the float noise buffer stays ~10 MB.
NOISE_ROW_CHUNK = 256


@dataclass
class PhoneCaptureSimulationSettings:
    """Share of samples simulated as phone captures, and the ranges drawn for each one.

    Attributes:
        probability: share of samples (train and val) cut at a phone-like crop scale.
        crop_scale_range: uniform range of the target crop scale (crop px per photo px).
        jpeg_probability: share of simulated samples re-encoded as JPEG.
        jpeg_quality_range: inclusive JPEG quality range.
        noise_probability: share of simulated samples given Gaussian sensor noise.
        noise_sigma_range: per-photo-pixel noise standard deviation, sRGB [0, 1] units (the
            pre-shrink averages it down by its factor, as on a real high-resolution photo).
    """

    probability: float = 0.5
    crop_scale_range: tuple[float, float] = (0.15, 0.6)
    jpeg_probability: float = 0.5
    jpeg_quality_range: tuple[int, int] = (70, 95)
    noise_probability: float = 0.5
    noise_sigma_range: tuple[float, float] = (0.003, 0.02)


def compute_crop_footprint_region(
    photo_to_eye_crop_affine: np.ndarray, crop_width: int, crop_height: int, photo_width: int, photo_height: int
) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom), exclusive right/bottom, of the photo pixels the crop reads, clipped to the photo."""
    crop_to_photo = cv2.invertAffineTransform(photo_to_eye_crop_affine)
    crop_corners = np.array([[-1.0, -1.0, 1.0], [crop_width, -1.0, 1.0], [-1.0, crop_height, 1.0], [crop_width, crop_height, 1.0]])
    photo_corners = crop_corners @ crop_to_photo.T
    left = max(0, math.floor(photo_corners[:, 0].min()) - REGION_MARGIN_PHOTO_PIXELS)
    top = max(0, math.floor(photo_corners[:, 1].min()) - REGION_MARGIN_PHOTO_PIXELS)
    right = min(photo_width, math.ceil(photo_corners[:, 0].max()) + REGION_MARGIN_PHOTO_PIXELS + 1)
    bottom = min(photo_height, math.ceil(photo_corners[:, 1].max()) + REGION_MARGIN_PHOTO_PIXELS + 1)
    return left, top, right, bottom


def convert_unit_float_to_uint8(unit_float_image: np.ndarray) -> np.ndarray:
    """[0, 1] float image -> rounded uint8 0-255 (contiguous copy)."""
    return np.clip(unit_float_image * UINT8_MAX + 0.5, 0, UINT8_MAX).astype(np.uint8)


def map_point_into_enlarged_region(point_xy: np.ndarray, region_left_top: tuple[int, int], enlarge_factor: float) -> np.ndarray:
    """Photo pixel -> enlarged-region pixel, using cv2.resize's pixel-center convention."""
    offset_point = np.asarray(point_xy, dtype=np.float64) - np.asarray(region_left_top, dtype=np.float64)
    return (offset_point + 0.5) * enlarge_factor - 0.5


def enlarge_crop_region_to_target_scale(
    photo_rgb: np.ndarray,
    lens_label_planes: np.ndarray,
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
    crop_width: int,
    crop_height: int,
    target_crop_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cut the crop's photo region and enlarge it so the same eyes give a crop at `target_crop_scale`.

    Args:
        photo_rgb: (H, W, 3) float32 sRGB in [0, 1].
        lens_label_planes: (H, W, 3) one-hot lens planes in [0, 1] (`lens_labels_to_one_hot_planes`).
    Returns:
        (photo region, label-plane region, left eye, right eye); both regions uint8 (planes 0-255),
        eyes in the region's pixels. The region is
        always cut (so degradation never touches the whole photo); it is enlarged only when the
        native scale is above the target, otherwise the crop keeps its native scale.
    """
    photo_to_crop = compute_photo_to_eye_crop_affine(image_left_eye_xy, image_right_eye_xy, crop_width, crop_height)
    enlarge_factor = max(1.0, eye_crop_scale(photo_to_crop) / target_crop_scale)
    photo_height, photo_width = photo_rgb.shape[:2]
    left, top, right, bottom = compute_crop_footprint_region(photo_to_crop, crop_width, crop_height, photo_width, photo_height)
    photo_region = convert_unit_float_to_uint8(photo_rgb[top:bottom, left:right])
    plane_region = convert_unit_float_to_uint8(lens_label_planes[top:bottom, left:right])
    if enlarge_factor > 1.0:
        # uint8 resize saturates, so cubic overshoot is clipped for free.
        photo_region = cv2.resize(photo_region, None, fx=enlarge_factor, fy=enlarge_factor, interpolation=cv2.INTER_CUBIC)
        plane_region = cv2.resize(plane_region, None, fx=enlarge_factor, fy=enlarge_factor, interpolation=cv2.INTER_LINEAR)
    return (
        photo_region,
        plane_region,
        map_point_into_enlarged_region(image_left_eye_xy, (left, top), enlarge_factor),
        map_point_into_enlarged_region(image_right_eye_xy, (left, top), enlarge_factor),
    )


def apply_capture_degradation(
    photo_rgb_uint8: np.ndarray, capture_settings: PhoneCaptureSimulationSettings, random_generator: np.random.Generator
) -> np.ndarray:
    """Optionally add Gaussian sensor noise, then optionally JPEG re-encode; (H, W, 3) uint8 RGB in and out."""
    degraded = photo_rgb_uint8
    if random_generator.random() < capture_settings.noise_probability:
        noise_sigma_levels = np.float32(random_generator.uniform(*capture_settings.noise_sigma_range) * UINT8_MAX)
        degraded = np.empty_like(photo_rgb_uint8)
        for row_start in range(0, photo_rgb_uint8.shape[0], NOISE_ROW_CHUNK):
            row_block = photo_rgb_uint8[row_start : row_start + NOISE_ROW_CHUNK].astype(np.float32)
            row_block += noise_sigma_levels * random_generator.standard_normal(size=row_block.shape, dtype=np.float32)
            degraded[row_start : row_start + NOISE_ROW_CHUNK] = np.clip(row_block + 0.5, 0, UINT8_MAX).astype(np.uint8)
    if random_generator.random() < capture_settings.jpeg_probability:
        quality = int(random_generator.integers(capture_settings.jpeg_quality_range[0], capture_settings.jpeg_quality_range[1] + 1))
        _, jpeg_bytes = cv2.imencode(".jpg", cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        degraded = cv2.cvtColor(cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return degraded
