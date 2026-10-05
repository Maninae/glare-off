"""Score how much REAL glare a source photo's lenses already show (0 = clean lens, higher = more glare).

Training on photos that already carry lens glare as "clean" targets teaches the model to keep real
glare. Two thresholds, because the two uses want opposite errors:

- score <= SOURCE_GLARE_SCORE_THRESHOLD: trainable (a low, conservative cut; losing clean faces is cheap).
- score >  REAL_GLARE_EVAL_SCORE_THRESHOLD: the real-glare evaluation set (a high bar, so it is mostly real glare).
- in between: ambiguous, used for neither.

Pure numpy/OpenCV on the photo's pixels inside each lens.

Two kinds of glare, two features (each per lens; the face takes its worse lens):

- **Bright blobs** (window, lamp, ring light, flash): pixels clearly brighter than the skin around
  the lens AND unlike skin in colour (Lab chroma far from this face's skin chroma, or much weaker
  than it). Brightness alone is not enough: lit under-eye skin is bright and, on pale skin, barely
  saturated, but it keeps the skin's chroma.
  Inside the eye opening only CLIPPED pixels count, because sclera is bright and unsaturated in every
  clean photo but rarely clips, while a reflection over the eye often does. Components smaller than
  MINIMUM_BLOB_AREA (several times a catchlight) are ignored as speckle.
- **Veil** (a soft wash over the whole lens): the lens interior (eye excluded) is lighter than the
  lit skin just outside the frame (an upper percentile of the ring, so frame, brows, hair, and
  shadows in the ring do not drag the reference down). Without glare a lens shows the same skin,
  slightly darker, so a positive lift is the signature of added light.

score = blob_fraction + max(0, veil_lift - VEIL_LIFT_ALLOWANCE)
The constants and the threshold were chosen by looking at contact sheets sorted by score (CLAUDE.md).
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Lens is eroded by this fraction of eye distance so frame pixels and rim shading never count.
LENS_EROSION_IN_EYE_DISTANCES = 0.03
# Skin ring around the lens: between these two dilation radii (skips the frame itself).
SKIN_RING_INNER_RADIUS_IN_EYE_DISTANCES = 0.10
SKIN_RING_OUTER_RADIUS_IN_EYE_DISTANCES = 0.22
# Eye opening (only clipped pixels count as blobs there; excluded from the veil): an ellipse with these semi-axes around each eye center.
EYE_OPENING_HALF_WIDTH_IN_EYE_DISTANCES = 0.24
EYE_OPENING_HALF_HEIGHT_IN_EYE_DISTANCES = 0.13
# Percentile of ring luma taken as "lit skin" (robust to dark frame, brow, and hair pixels in the ring).
SKIN_RING_REFERENCE_PERCENTILE = 70
# A blob pixel is at least this much brighter (luma, 0..1) than the skin reference ...
BLOB_LUMINANCE_EXCESS = 0.10
# ... and either its Lab (a, b) is at least this far from the skin's median (a, b) ...
BLOB_MINIMUM_CHROMA_DISTANCE_FROM_SKIN = 10.0
# ... or its chroma magnitude is below this fraction of the skin's (whitish glare on coloured skin).
BLOB_MAXIMUM_CHROMA_FRACTION_OF_SKIN = 0.45
# Bright components smaller than this (in eye distance squared) are speckle, not glare.
MINIMUM_BLOB_AREA_IN_EYE_DISTANCE_SQUARED = 0.003
# Max channel value (0..1) at or above which a pixel counts as clipped.
CLIPPED_CHANNEL_VALUE = 0.95
MINIMUM_LENS_PIXEL_COUNT = 50
VEIL_LIFT_ALLOWANCE = 0.02
# Chosen on review sheets of 20 faces per score band (CLAUDE.md): about half the faces at 0.02-0.035
# still show some glare, and about 70% at 0.1-0.2 show clear glare.
SOURCE_GLARE_SCORE_THRESHOLD = 0.02
REAL_GLARE_EVAL_SCORE_THRESHOLD = 0.08


@dataclass
class LensGlareFeatures:
    """Glare evidence measured on one lens."""

    blob_fraction: float
    veil_lift: float
    lens_median_luminance: float
    skin_ring_reference_luminance: float


def rgb_to_luminance(photo_rgb_float: np.ndarray) -> np.ndarray:
    """Return Rec. 709 luma of sRGB values in [0, 1] (perceptual brightness, no linearization)."""
    return photo_rgb_float @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def rgb_to_lab_chroma(photo_rgb_float: np.ndarray) -> np.ndarray:
    """Return the CIE Lab (a, b) chroma plane, shape (H, W, 2), from sRGB in [0, 1]."""
    return cv2.cvtColor(photo_rgb_float, cv2.COLOR_RGB2Lab)[:, :, 1:]


def disk_kernel(radius_pixels: float) -> np.ndarray:
    """Return an elliptical structuring element of the given radius (at least 1 px)."""
    radius_pixels = max(1, int(round(radius_pixels)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_pixels + 1, 2 * radius_pixels + 1))


def remove_components_smaller_than(binary_mask: np.ndarray, minimum_area_pixels: float) -> np.ndarray:
    """Return the mask with connected components below `minimum_area_pixels` removed."""
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    kept_labels = [label for label in range(1, component_count) if component_stats[label, cv2.CC_STAT_AREA] >= minimum_area_pixels]
    return np.isin(component_labels, kept_labels)


def build_eye_opening_mask(image_shape: tuple[int, int], eye_centers_xy: list[np.ndarray], eye_distance: float) -> np.ndarray:
    """Return a bool (H, W) mask of an ellipse around each eye center, tilted with the eye line."""
    eye_opening_mask = np.zeros(image_shape, dtype=np.uint8)
    first_eye, second_eye = (np.asarray(eye_xy, dtype=np.float64) for eye_xy in eye_centers_xy)
    roll_angle_degrees = float(np.degrees(np.arctan2(second_eye[1] - first_eye[1], second_eye[0] - first_eye[0])))
    semi_axes = (
        int(round(EYE_OPENING_HALF_WIDTH_IN_EYE_DISTANCES * eye_distance)),
        int(round(EYE_OPENING_HALF_HEIGHT_IN_EYE_DISTANCES * eye_distance)),
    )
    for eye_xy in (first_eye, second_eye):
        cv2.ellipse(eye_opening_mask, (int(round(eye_xy[0])), int(round(eye_xy[1]))), semi_axes, roll_angle_degrees, 0, 360, 1, thickness=cv2.FILLED)
    return eye_opening_mask.astype(bool)


def measure_lens_glare_features(
    photo_rgb_float: np.ndarray,
    single_lens_mask: np.ndarray,
    all_lenses_mask: np.ndarray,
    eye_opening_mask: np.ndarray,
    eye_distance: float,
) -> LensGlareFeatures | None:
    """Measure blob and veil evidence on one lens; None when the judged area is too small.

    Args:
        photo_rgb_float: (H, W, 3) sRGB in [0, 1].
        single_lens_mask: bool (H, W), this lens only.
        all_lenses_mask: bool (H, W), both lenses (excluded from the skin ring).
        eye_opening_mask: bool (H, W), eye openings to ignore.
        eye_distance: in pixels; sets every size threshold.
    """
    eroded_lens = cv2.erode(single_lens_mask.astype(np.uint8), disk_kernel(LENS_EROSION_IN_EYE_DISTANCES * eye_distance)).astype(bool)
    judged_lens_pixels = eroded_lens & ~eye_opening_mask
    if judged_lens_pixels.sum() < MINIMUM_LENS_PIXEL_COUNT:
        return None
    luminance = rgb_to_luminance(photo_rgb_float)
    lens_uint8 = single_lens_mask.astype(np.uint8)
    outer_ring = cv2.dilate(lens_uint8, disk_kernel(SKIN_RING_OUTER_RADIUS_IN_EYE_DISTANCES * eye_distance)).astype(bool)
    inner_ring = cv2.dilate(lens_uint8, disk_kernel(SKIN_RING_INNER_RADIUS_IN_EYE_DISTANCES * eye_distance)).astype(bool)
    skin_ring = outer_ring & ~inner_ring & ~all_lenses_mask & ~eye_opening_mask
    lens_median_luminance = float(np.median(luminance[judged_lens_pixels]))
    if skin_ring.any():
        skin_ring_reference_luminance = float(np.percentile(luminance[skin_ring], SKIN_RING_REFERENCE_PERCENTILE))
    else:
        skin_ring_reference_luminance = lens_median_luminance
    skin_reference_luminance = max(lens_median_luminance, skin_ring_reference_luminance)
    lab_chroma = rgb_to_lab_chroma(photo_rgb_float)
    skin_chroma = np.median(lab_chroma[judged_lens_pixels], axis=0)
    chroma_distance_from_skin = np.linalg.norm(lab_chroma - skin_chroma, axis=2)
    chroma_magnitude = np.linalg.norm(lab_chroma, axis=2)
    unlike_skin_colour = (chroma_distance_from_skin >= BLOB_MINIMUM_CHROMA_DISTANCE_FROM_SKIN) | (
        chroma_magnitude <= BLOB_MAXIMUM_CHROMA_FRACTION_OF_SKIN * float(np.linalg.norm(skin_chroma))
    )
    bright_unlike_skin = (luminance > skin_reference_luminance + BLOB_LUMINANCE_EXCESS) & unlike_skin_colour
    clipped_over_eye = (photo_rgb_float.max(axis=2) >= CLIPPED_CHANNEL_VALUE) & eroded_lens & eye_opening_mask
    blob_pixels = remove_components_smaller_than(
        (bright_unlike_skin & judged_lens_pixels) | clipped_over_eye, MINIMUM_BLOB_AREA_IN_EYE_DISTANCE_SQUARED * eye_distance**2
    )
    return LensGlareFeatures(
        blob_fraction=float(blob_pixels.sum() / eroded_lens.sum()),
        veil_lift=lens_median_luminance - skin_ring_reference_luminance,
        lens_median_luminance=lens_median_luminance,
        skin_ring_reference_luminance=skin_ring_reference_luminance,
    )


def combine_lens_features_into_glare_score(lens_features: LensGlareFeatures) -> float:
    """Return one lens's glare score from its features (formula in the module docstring)."""
    return lens_features.blob_fraction + max(0.0, lens_features.veil_lift - VEIL_LIFT_ALLOWANCE)


def compute_source_glare_score(
    photo_rgb_uint8: np.ndarray,
    lens_label_map: np.ndarray,
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
) -> tuple[float, list[LensGlareFeatures]]:
    """Return (face glare score = worst lens, per-lens features) for a photo and its lens label map.

    Returns score 0.0 with no features when no lens is large enough to judge.
    """
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_distance = float(np.hypot(*(right_eye - left_eye)))
    lens_rows, lens_columns = np.nonzero(lens_label_map)
    if lens_rows.size == 0:
        return 0.0, []
    # Work on the lens bounding box plus the skin-ring reach: same result, ~10x less pixel work.
    margin = int(np.ceil(SKIN_RING_OUTER_RADIUS_IN_EYE_DISTANCES * eye_distance)) + 2
    top, bottom = max(0, lens_rows.min() - margin), min(lens_label_map.shape[0], lens_rows.max() + margin + 1)
    left, right = max(0, lens_columns.min() - margin), min(lens_label_map.shape[1], lens_columns.max() + margin + 1)
    lens_label_map = lens_label_map[top:bottom, left:right]
    photo_rgb_float = photo_rgb_uint8[top:bottom, left:right].astype(np.float32) / 255.0
    crop_offset_xy = np.array([left, top], dtype=np.float64)
    eye_opening_mask = build_eye_opening_mask(lens_label_map.shape, [left_eye - crop_offset_xy, right_eye - crop_offset_xy], eye_distance)
    all_lenses_mask = lens_label_map > 0
    per_lens_features = []
    for lens_value in np.unique(lens_label_map[all_lenses_mask]):
        lens_features = measure_lens_glare_features(
            photo_rgb_float, lens_label_map == lens_value, all_lenses_mask, eye_opening_mask, eye_distance
        )
        if lens_features is not None:
            per_lens_features.append(lens_features)
    if not per_lens_features:
        return 0.0, []
    return max(combine_lens_features_into_glare_score(features) for features in per_lens_features), per_lens_features
