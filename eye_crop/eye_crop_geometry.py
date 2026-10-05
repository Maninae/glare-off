"""Eye-crop geometry: the similarity transform between a photo and the aligned glasses crop.

This is THE definition of the crop the glare model sees. Training data, evaluation, and the
browser app (`app/js/eye_crop_geometry.js`, a line-for-line port) must all produce the same
crop from the same two eye centers, or the model is run on framing it never trained on.

- The crop is EYE_CROP_WIDTH x EYE_CROP_HEIGHT, eyes level, eye midpoint at the crop center.
- The distance between the eye centers spans EYE_DISTANCE_FRACTION of the crop width.
- Pure numpy/OpenCV, no detector here: callers pass eye centers (see `yunet_eye_detector.py`).
"""

import cv2
import numpy as np

EYE_CROP_WIDTH = 512
EYE_CROP_HEIGHT = 256
# Eye-to-eye distance as a fraction of crop width; 0.36 leaves room for wide frames and temples.
EYE_DISTANCE_FRACTION = 0.36


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

    - Use INTER_AREA when the crop shrinks the photo (the usual case for phone photos) and
      INTER_CUBIC when it enlarges; callers that care pick via `eye_crop_scale`.
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
