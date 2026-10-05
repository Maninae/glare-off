"""Eye-crop geometry: the similarity transform between a photo and the aligned glasses crop.

This is THE definition of the crop the glare model sees. Training data, evaluation, and the
browser app (`app/js/eye_crop_geometry.js`, a line-for-line port) must all produce the same
crop from the same two eye centers, or the model is run on framing it never trained on.

- The crop is EYE_CROP_WIDTH x EYE_CROP_HEIGHT, eyes level, eye midpoint at the crop center.
- The distance between the eye centers spans EYE_DISTANCE_FRACTION of the crop width.
- Pure numpy/OpenCV, no detector here: callers pass eye centers (see `yunet_eye_detector.py`).
- `extract_eye_crop_with_area_prefilter` is the crop resampling every pixel producer should use
  (training and the app): bilinear, with an integer INTER_AREA pre-shrink when the crop would
  shrink the photo below `AREA_PREFILTER_SCALE_THRESHOLD` (bilinear alone aliases there).
"""

import math

import cv2
import numpy as np

EYE_CROP_WIDTH = 512
EYE_CROP_HEIGHT = 256
# Eye-to-eye distance as a fraction of crop width; 0.36 leaves room for wide frames and temples.
EYE_DISTANCE_FRACTION = 0.36
# Below this crop scale a plain bilinear warp skips photo pixels and aliases; pre-shrink first.
AREA_PREFILTER_SCALE_THRESHOLD = 0.5


def compute_photo_to_eye_crop_affine(
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
    crop_width: int = EYE_CROP_WIDTH,
    crop_height: int = EYE_CROP_HEIGHT,
) -> np.ndarray:
    """Return the 2x3 affine mapping photo pixel coordinates to eye-crop pixel coordinates.

    Args:
        image_left_eye_xy: (x, y) of the eye that appears on the LEFT side of the image.
        image_right_eye_xy: (x, y) of the eye that appears on the RIGHT side of the image.
        crop_width, crop_height: output crop size; pass 2x values for a high-resolution pass.
    Returns:
        float64 array of shape (2, 3): rotation + uniform scale + translation.

    - Eye order is by image position, not anatomy, so the crop is never flipped upside down.
    """
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.hypot(eye_vector[0], eye_vector[1]))
    if eye_distance < 1e-6:
        raise ValueError("eye centers coincide; cannot build an eye crop")
    roll_angle_degrees = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
    scale = (crop_width * EYE_DISTANCE_FRACTION) / eye_distance
    eye_midpoint = (left_eye + right_eye) / 2.0
    affine = cv2.getRotationMatrix2D((float(eye_midpoint[0]), float(eye_midpoint[1])), roll_angle_degrees, scale)
    affine[0, 2] += crop_width / 2.0 - eye_midpoint[0]
    affine[1, 2] += crop_height / 2.0 - eye_midpoint[1]
    return affine


def extract_eye_crop(
    photo: np.ndarray,
    photo_to_eye_crop_affine: np.ndarray,
    crop_width: int = EYE_CROP_WIDTH,
    crop_height: int = EYE_CROP_HEIGHT,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Warp the photo into the aligned eye crop (reflect-padded where the crop leaves the photo).

    - warpAffine has no area filter: INTER_AREA here behaves as bilinear, with no prefilter. For
      image pixels the model sees, use `extract_eye_crop_with_area_prefilter` (what the app does).
      This plain warp stays for label maps (INTER_NEAREST) and the app parity fixtures.
    """
    return cv2.warpAffine(
        photo,
        photo_to_eye_crop_affine,
        (crop_width, crop_height),
        flags=interpolation,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def eye_crop_scale(photo_to_eye_crop_affine: np.ndarray) -> float:
    """Return crop pixels per photo pixel (below 1 means the crop is a downscale of the photo)."""
    return float(np.hypot(photo_to_eye_crop_affine[0, 0], photo_to_eye_crop_affine[0, 1]))


def compute_area_prefilter_factor(crop_scale: float) -> int:
    """Return the integer INTER_AREA shrink factor applied before the warp (1 = no pre-shrink).

    For crop_scale < AREA_PREFILTER_SCALE_THRESHOLD the factor is floor(1 / crop_scale), so the
    remaining warp scale `crop_scale * factor` lands in (1 - crop_scale, 1], always inside [0.5, 1].
    """
    if crop_scale >= AREA_PREFILTER_SCALE_THRESHOLD:
        return 1
    # The epsilon keeps 1/0.25 = 3.9999... from flooring to 3.
    return max(2, math.floor(1.0 / crop_scale + 1e-9))


def compose_affine_with_area_prefilter(photo_to_eye_crop_affine: np.ndarray, prefilter_factor: int) -> np.ndarray:
    """Return the affine from the pre-shrunk photo's pixels to the same crop pixels.

    `cv2.resize(fx=1/n)` maps pixel centers as `x_small = (x + 0.5) / n - 0.5`, so a
    pre-shrunk pixel sits at photo `x = n * x_small + (n - 1) / 2`.
    """
    factor = float(prefilter_factor)
    small_to_photo = np.array([[factor, 0.0, (factor - 1.0) / 2.0], [0.0, factor, (factor - 1.0) / 2.0], [0.0, 0.0, 1.0]])
    return photo_to_eye_crop_affine @ small_to_photo


def extract_eye_crop_with_area_prefilter(
    photo: np.ndarray,
    photo_to_eye_crop_affine: np.ndarray,
    crop_width: int = EYE_CROP_WIDTH,
    crop_height: int = EYE_CROP_HEIGHT,
) -> np.ndarray:
    """Warp the photo into the eye crop the way the app does: optional integer area pre-shrink, then bilinear.

    Args:
        photo: (H, W) or (H, W, C) image; float or uint8.
        photo_to_eye_crop_affine: from `compute_photo_to_eye_crop_affine` (unchanged by the pre-shrink).
    Returns:
        (crop_height, crop_width[, C]) crop in the same crop coordinates as `extract_eye_crop`, so
        `warp_eye_crop_layer_back_to_photo` with the ORIGINAL affine still round-trips.

    - Scale >= 0.5: one bilinear warpAffine (upscales too: never cubic, the app is bilinear).
    - Scale < 0.5: `cv2.resize(photo, fx=fy=1/n, INTER_AREA)` with n from `compute_area_prefilter_factor`
      (n x n box average, blocks aligned to the photo origin), then bilinear with the composed affine.
    """
    prefilter_factor = compute_area_prefilter_factor(eye_crop_scale(photo_to_eye_crop_affine))
    if prefilter_factor > 1:
        photo = cv2.resize(photo, None, fx=1.0 / prefilter_factor, fy=1.0 / prefilter_factor, interpolation=cv2.INTER_AREA)
        photo_to_eye_crop_affine = compose_affine_with_area_prefilter(photo_to_eye_crop_affine, prefilter_factor)
    return cv2.warpAffine(
        photo,
        photo_to_eye_crop_affine,
        (crop_width, crop_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def warp_eye_crop_layer_back_to_photo(
    eye_crop_layer: np.ndarray,
    photo_to_eye_crop_affine: np.ndarray,
    photo_width: int,
    photo_height: int,
) -> np.ndarray:
    """Warp a crop-space layer (a delta image or a mask) back onto the photo's pixel grid.

    Pixels the crop does not cover come back as zero, so a delta layer adds nothing there and
    a mask layer blends nothing there.
    """
    return cv2.warpAffine(
        eye_crop_layer,
        photo_to_eye_crop_affine,
        (photo_width, photo_height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
