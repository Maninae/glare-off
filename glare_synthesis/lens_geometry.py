"""Per-lens geometry derived from the crop's lens label mask.

The renderer needs, for each lens present: where it is (centroid, box, size), the lens interior
(with polygon corners rounded off so an imperfect mask never prints its outline into the glare),
and a normalized distance-to-rim map for the edge falloff. The per-sample soft edge (random
erosion + feather) is built at render time by `build_soft_lens_mask`.
Label convention (source manifest contract): 0 background, 1 image-left lens, 2 image-right lens.
"""

from dataclasses import dataclass

import cv2
import numpy as np

IMAGE_LEFT_LENS_LABEL = 1
IMAGE_RIGHT_LENS_LABEL = 2
# A lens smaller than this many pixels (at any crop size) is treated as absent.
MINIMUM_LENS_AREA_PIXELS = 40
# Gaussian used to round polygon corners of hand-drawn or segmenter masks (pixels at 512 wide).
MASK_CORNER_ROUNDING_SIGMA_AT_512 = 2.0
REFERENCE_CROP_WIDTH = 512


@dataclass
class LensGeometry:
    """One lens in crop space; full-crop-size arrays plus scalar shape summaries."""

    lens_label: int
    lens_pixels: np.ndarray
    rim_distance_normalized: np.ndarray
    center_xy: np.ndarray
    width_pixels: float
    height_pixels: float
    bounding_box_xyxy: tuple[int, int, int, int]


def round_mask_corners(lens_pixels: np.ndarray, crop_width: int) -> np.ndarray:
    """Blur-and-threshold the mask to round its corners, never growing it past the original."""
    rounding_sigma = MASK_CORNER_ROUNDING_SIGMA_AT_512 * crop_width / REFERENCE_CROP_WIDTH
    rounded = cv2.GaussianBlur(lens_pixels.astype(np.float32), (0, 0), rounding_sigma) > 0.5
    return rounded & lens_pixels


def build_soft_lens_mask(lens_pixels_region: np.ndarray, erosion_pixels: float, feather_sigma_pixels: float) -> np.ndarray:
    """Soft interior mask for one sample: erode by `erosion_pixels`, feather, clamp to the lens.

    The clamp keeps every soft value inside the lens, so glare ends at the rim with a soft edge of
    about 2 x feather sigma and never paints onto the frame.
    """
    lens_uint8 = lens_pixels_region.astype(np.uint8)
    erosion_radius = int(round(erosion_pixels))
    if erosion_radius > 0:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_radius + 1, 2 * erosion_radius + 1))
        lens_uint8 = cv2.erode(lens_uint8, erosion_kernel)
    soft_mask = cv2.GaussianBlur(lens_uint8.astype(np.float32), (0, 0), max(feather_sigma_pixels, 0.3))
    return soft_mask * lens_pixels_region


def analyze_lens_label_mask(lens_label_mask: np.ndarray) -> list[LensGeometry]:
    """Return a `LensGeometry` for every lens present in the label mask, image-left first.

    - Lenses below MINIMUM_LENS_AREA_PIXELS are skipped, so a returned empty list means "no glasses".
    - Width/height are the lens's axis-aligned box in crop pixels (the crop is eye-level aligned).
    """
    if lens_label_mask.ndim != 2:
        raise ValueError(f"lens_label_mask must be (H, W), got shape {lens_label_mask.shape}")
    crop_width = lens_label_mask.shape[1]
    lens_geometries = []
    for lens_label in (IMAGE_LEFT_LENS_LABEL, IMAGE_RIGHT_LENS_LABEL):
        raw_lens_pixels = lens_label_mask == lens_label
        if int(raw_lens_pixels.sum()) < MINIMUM_LENS_AREA_PIXELS:
            continue
        lens_pixels = round_mask_corners(raw_lens_pixels, crop_width)
        if int(lens_pixels.sum()) < MINIMUM_LENS_AREA_PIXELS:
            lens_pixels = raw_lens_pixels
        rows, columns = np.nonzero(lens_pixels)
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        rim_distance = cv2.distanceTransform(lens_pixels.astype(np.uint8), cv2.DIST_L2, 3)
        # Normalize by the lens half-height so 1.0 is roughly "as deep inside as the lens gets".
        rim_distance_normalized = np.clip(rim_distance / max((y1 - y0) / 2.0, 1.0), 0.0, 1.0).astype(np.float32)
        lens_geometries.append(
            LensGeometry(
                lens_label=lens_label,
                lens_pixels=lens_pixels.astype(np.float32),
                rim_distance_normalized=rim_distance_normalized,
                center_xy=np.array([columns.mean(), rows.mean()], dtype=np.float64),
                width_pixels=float(x1 - x0),
                height_pixels=float(y1 - y0),
                bounding_box_xyxy=(x0, y0, x1, y1),
            )
        )
    return lens_geometries
