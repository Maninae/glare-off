"""Turn a binary lens segmentation into the contract's label map: 1 = image-left lens, 2 = image-right lens.

Pure numpy/OpenCV, no model. The steps:

1. Cut the binary mask along the perpendicular bisector of the two eye centers, so lenses the
   segmenter merged across the nose bridge separate cleanly.
2. On each side, keep the connected component closest to that side's eye (containing it, ideally)
   and fill its holes (eye catchlights sometimes punch holes in the segmentation).
3. Check each lens is plausible relative to eye distance `d`: area, width, eye inside or near it,
   and solidity (area / convex hull area): real lenses are convex, leaks onto skin or background are not.
4. Two plausible lenses -> TWO_LENSES. One plausible lens is accepted as SINGLE_LENS only when the
   caller says the head is turned strongly (the far lens is then legitimately hidden or tiny).
"""

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class LensMaskStatus(str, Enum):
    """Outcome of the left/right lens assignment for one face."""

    TWO_LENSES = "two_lenses"
    SINGLE_LENS = "single_lens"
    FAILED = "failed"


class LensLabel(int, Enum):
    """Pixel values of the lens-mask PNG (the source-manifest contract)."""

    BACKGROUND = 0
    IMAGE_LEFT_LENS = 1
    IMAGE_RIGHT_LENS = 2


# Plausibility bounds, in units of eye distance d (areas in d squared). Typical frontal lens: about 0.8d x 0.5d.
MINIMUM_LENS_AREA_IN_EYE_DISTANCE_SQUARED = 0.08
MAXIMUM_LENS_AREA_IN_EYE_DISTANCE_SQUARED = 1.2
MINIMUM_LENS_WIDTH_IN_EYE_DISTANCES = 0.3
# The eye must be inside its lens or at most this far from the nearest lens pixel.
MAXIMUM_EYE_TO_LENS_DISTANCE_IN_EYE_DISTANCES = 0.15
# Lens area over its convex hull area; below this the mask has leaked (chosen on review sheets).
MINIMUM_LENS_SOLIDITY = 0.93
# Smaller-to-larger lens area ratio below which a frontal face's lens pair is implausible.
MINIMUM_FRONTAL_LENS_AREA_RATIO = 0.4


@dataclass
class LensAssignmentResult:
    """Label map plus the evidence used to accept or reject it."""

    lens_label_map: np.ndarray
    status: LensMaskStatus
    failure_reason: str = ""
    lens_area_in_eye_distance_squared: dict[str, float] = field(default_factory=dict)


def pixels_on_image_left_side_of_eye_bisector(
    image_height: int, image_width: int, image_left_eye_xy: np.ndarray, image_right_eye_xy: np.ndarray
) -> np.ndarray:
    """Return a bool (H, W) map: True where a pixel is on the image-left eye's side of the bisector."""
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_vector = right_eye - left_eye
    eye_midpoint = (left_eye + right_eye) / 2.0
    pixel_rows, pixel_columns = np.mgrid[0:image_height, 0:image_width]
    signed_projection = (pixel_columns - eye_midpoint[0]) * eye_vector[0] + (pixel_rows - eye_midpoint[1]) * eye_vector[1]
    return signed_projection < 0


def fill_component_holes(component_mask: np.ndarray) -> np.ndarray:
    """Return the component with every interior hole filled (outer contours drawn solid)."""
    outer_contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled_mask = np.zeros(component_mask.shape, dtype=np.uint8)
    cv2.drawContours(filled_mask, outer_contours, -1, 1, thickness=cv2.FILLED)
    return filled_mask.astype(bool)


def select_component_nearest_eye(side_mask: np.ndarray, eye_xy: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return (component mask nearest the eye, eye-to-component pixel distance), or (None, inf) if empty."""
    component_count, component_labels = cv2.connectedComponents(side_mask.astype(np.uint8), connectivity=8)
    if component_count <= 1:
        return None, float("inf")
    eye_column = int(np.clip(round(eye_xy[0]), 0, side_mask.shape[1] - 1))
    eye_row = int(np.clip(round(eye_xy[1]), 0, side_mask.shape[0] - 1))
    label_under_eye = component_labels[eye_row, eye_column]
    if label_under_eye > 0:
        return component_labels == label_under_eye, 0.0
    # Distance from the eye to every non-background pixel; pick the component owning the closest one.
    component_rows, component_columns = np.nonzero(component_labels)
    pixel_distances = np.hypot(component_columns - eye_xy[0], component_rows - eye_xy[1])
    closest_pixel_index = int(np.argmin(pixel_distances))
    closest_label = component_labels[component_rows[closest_pixel_index], component_columns[closest_pixel_index]]
    return component_labels == closest_label, float(pixel_distances[closest_pixel_index])


def measure_mask_solidity(binary_mask: np.ndarray) -> float:
    """Return area / convex hull area of a mask's largest outer contour (1.0 for a convex blob)."""
    outer_contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(outer_contours, key=cv2.contourArea)
    hull_area = cv2.contourArea(cv2.convexHull(largest_contour))
    return float(cv2.contourArea(largest_contour) / hull_area) if hull_area > 0 else 0.0


def check_lens_plausibility(lens_mask: np.ndarray | None, eye_to_lens_distance: float, eye_distance: float) -> tuple[bool, str, float]:
    """Return (is plausible, reason if not, area in d squared) for one candidate lens."""
    if lens_mask is None:
        return False, "no_component", 0.0
    lens_area = float(lens_mask.sum()) / eye_distance**2
    if eye_to_lens_distance > MAXIMUM_EYE_TO_LENS_DISTANCE_IN_EYE_DISTANCES * eye_distance:
        return False, "eye_outside_lens", lens_area
    if lens_area < MINIMUM_LENS_AREA_IN_EYE_DISTANCE_SQUARED:
        return False, "lens_too_small", lens_area
    if lens_area > MAXIMUM_LENS_AREA_IN_EYE_DISTANCE_SQUARED:
        return False, "lens_too_large", lens_area
    if measure_mask_solidity(lens_mask) < MINIMUM_LENS_SOLIDITY:
        return False, "lens_not_convex", lens_area
    lens_columns = np.nonzero(lens_mask.any(axis=0))[0]
    if (lens_columns.max() - lens_columns.min() + 1) < MINIMUM_LENS_WIDTH_IN_EYE_DISTANCES * eye_distance:
        return False, "lens_too_narrow", lens_area
    return True, "", lens_area


def assign_left_right_lenses(
    lens_binary_mask: np.ndarray,
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
    head_is_strongly_turned: bool,
) -> LensAssignmentResult:
    """Split a binary lens mask into the contract's left/right label map and judge its plausibility.

    Args:
        lens_binary_mask: bool (H, W), True where the segmenter says lens.
        image_left_eye_xy, image_right_eye_xy: eye centers ordered by image x.
        head_is_strongly_turned: allow a single lens (the far one hidden by a turned head).
    Returns:
        LensAssignmentResult; the label map is all background when the status is FAILED.
    """
    image_height, image_width = lens_binary_mask.shape
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_distance = float(np.hypot(*(right_eye - left_eye)))
    left_side = pixels_on_image_left_side_of_eye_bisector(image_height, image_width, left_eye, right_eye)
    lens_label_map = np.zeros((image_height, image_width), dtype=np.uint8)
    plausible_labels, failure_reasons, lens_areas = [], [], {}
    for lens_label, side_pixels, eye_xy in (
        (LensLabel.IMAGE_LEFT_LENS, left_side, left_eye),
        (LensLabel.IMAGE_RIGHT_LENS, ~left_side, right_eye),
    ):
        lens_component, eye_to_lens_distance = select_component_nearest_eye(lens_binary_mask & side_pixels, eye_xy)
        if lens_component is not None:
            lens_component = fill_component_holes(lens_component) & side_pixels
        is_plausible, failure_reason, lens_area = check_lens_plausibility(lens_component, eye_to_lens_distance, eye_distance)
        lens_areas[lens_label.name.lower()] = round(lens_area, 4)
        if is_plausible:
            lens_label_map[lens_component] = lens_label.value
            plausible_labels.append(lens_label)
        else:
            failure_reasons.append(f"{lens_label.name.lower()}:{failure_reason}")
    if len(plausible_labels) == 2:
        area_ratio = min(lens_areas.values()) / max(lens_areas.values())
        if area_ratio < MINIMUM_FRONTAL_LENS_AREA_RATIO and not head_is_strongly_turned:
            return LensAssignmentResult(np.zeros_like(lens_label_map), LensMaskStatus.FAILED, "asymmetric_lens_areas", lens_areas)
        return LensAssignmentResult(lens_label_map, LensMaskStatus.TWO_LENSES, "", lens_areas)
    if len(plausible_labels) == 1 and head_is_strongly_turned:
        return LensAssignmentResult(lens_label_map, LensMaskStatus.SINGLE_LENS, ";".join(failure_reasons), lens_areas)
    return LensAssignmentResult(np.zeros_like(lens_label_map), LensMaskStatus.FAILED, ";".join(failure_reasons), lens_areas)
