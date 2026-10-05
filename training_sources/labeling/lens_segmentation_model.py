"""Lens segmentation (glasses lens interior, not the frame) as an offline labeling aid.

The network is the `lenses` LR-ASPP MobileNetV3-Large from mantasu/glasses-detector (MIT code,
release v1.0.0). We do not install or run that package: the architecture is torchvision's
`lraspp_mobilenet_v3_large(num_classes=1)` exactly as its source builds it, and only the weight
file is downloaded and loaded with `weights_only=True`.

Preprocessing matches the package's `predict`: RGB, resized to 256x256, ImageNet mean/std, and a
pixel is lens where the `out` logit is above 0.

- Input framing differs from the package on purpose: instead of the whole 1024 face, we feed an
  eye-level square crop LENS_CROP_SIDE_IN_EYE_DISTANCES eye distances wide. That doubles the
  resolution on the lenses and, checked by eye on a sample, fixes lenses merging across the bridge.
- Logits are averaged with the horizontally flipped crop's (cheap test-time augmentation).
- Never shipped: the weights' training data includes CelebAMask-HQ (non-commercial research).
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

from training_sources.training_sources_paths import LENS_SEGMENTATION_WEIGHTS_PATH

logger = logging.getLogger(__name__)

LENS_SEGMENTER_INPUT_SIDE = 256
IMAGENET_CHANNEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_CHANNEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Square crop side as a multiple of eye distance; wide enough for large frames on a frontal face.
LENS_CROP_SIDE_IN_EYE_DISTANCES = 3.0
# The crop centre sits slightly below the eye line because lenses extend further down than up.
LENS_CROP_CENTRE_DOWN_SHIFT_IN_EYE_DISTANCES = 0.1
# Logit given to photo pixels the segmenter never saw: confidently "not lens".
OUTSIDE_CROP_LOGIT = -20.0


def build_lens_segmentation_network(weights_path: Path = LENS_SEGMENTATION_WEIGHTS_PATH) -> torch.nn.Module:
    """Rebuild the LR-ASPP MobileNetV3-Large lens segmenter and load its weights (CPU, eval mode).

    - `weights_backbone=None` stops torchvision from downloading ImageNet backbone weights.
    - Strict loading: a missing or unexpected key raises, so an architecture mismatch fails loud.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"lens segmentation weights missing: {weights_path}")
    network = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=1)
    state_dict = torch.load(weights_path, weights_only=True, map_location="cpu")
    network.load_state_dict(state_dict, strict=True)
    network.eval()
    return network


def compute_photo_to_lens_crop_affine(image_left_eye_xy: np.ndarray, image_right_eye_xy: np.ndarray) -> np.ndarray:
    """Return the 2x3 similarity mapping photo pixels into the eye-level square segmenter input."""
    left_eye = np.asarray(image_left_eye_xy, dtype=np.float64)
    right_eye = np.asarray(image_right_eye_xy, dtype=np.float64)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.hypot(*eye_vector))
    roll_angle_degrees = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
    scale = LENS_SEGMENTER_INPUT_SIDE / (LENS_CROP_SIDE_IN_EYE_DISTANCES * eye_distance)
    eye_midpoint = (left_eye + right_eye) / 2.0
    # "Down" in the face frame is perpendicular to the eye line.
    face_down_direction = np.array([-eye_vector[1], eye_vector[0]]) / eye_distance
    crop_centre = eye_midpoint + face_down_direction * LENS_CROP_CENTRE_DOWN_SHIFT_IN_EYE_DISTANCES * eye_distance
    affine = cv2.getRotationMatrix2D((float(crop_centre[0]), float(crop_centre[1])), roll_angle_degrees, scale)
    affine[0, 2] += LENS_SEGMENTER_INPUT_SIDE / 2.0 - crop_centre[0]
    affine[1, 2] += LENS_SEGMENTER_INPUT_SIDE / 2.0 - crop_centre[1]
    return affine


def normalize_rgb_for_segmenter(crop_rgb_uint8: np.ndarray) -> torch.Tensor:
    """Turn a (256, 256, 3) uint8 RGB crop into a normalized (1, 3, 256, 256) float tensor."""
    crop_float = crop_rgb_uint8.astype(np.float32) / 255.0
    crop_normalized = (crop_float - IMAGENET_CHANNEL_MEAN) / IMAGENET_CHANNEL_STD
    return torch.from_numpy(np.ascontiguousarray(crop_normalized.transpose(2, 0, 1)))[None]


class LensSegmenter:
    """Holds the lens network and turns a photo plus eye centers into a lens logit map on the photo grid."""

    def __init__(self, weights_path: Path = LENS_SEGMENTATION_WEIGHTS_PATH):
        """Load the network once; reuse the instance for every photo."""
        self.network = build_lens_segmentation_network(weights_path)

    def predict_lens_logits_on_photo(
        self, photo_rgb_uint8: np.ndarray, image_left_eye_xy: np.ndarray, image_right_eye_xy: np.ndarray
    ) -> np.ndarray:
        """Return a float32 (H, W) map of lens logits on the photo's pixel grid (> 0 means lens).

        Pixels outside the segmenter's crop come back as a large negative logit (not lens).
        """
        photo_height, photo_width = photo_rgb_uint8.shape[:2]
        photo_to_crop_affine = compute_photo_to_lens_crop_affine(image_left_eye_xy, image_right_eye_xy)
        crop_rgb = cv2.warpAffine(
            photo_rgb_uint8,
            photo_to_crop_affine,
            (LENS_SEGMENTER_INPUT_SIDE, LENS_SEGMENTER_INPUT_SIDE),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        input_batch = torch.cat([normalize_rgb_for_segmenter(crop_rgb), normalize_rgb_for_segmenter(crop_rgb[:, ::-1])])
        with torch.inference_mode():
            output_logits = self.network(input_batch)["out"][:, 0].numpy()
        crop_logits = (output_logits[0] + output_logits[1][:, ::-1]) / 2.0
        return cv2.warpAffine(
            crop_logits.astype(np.float32),
            photo_to_crop_affine,
            (photo_width, photo_height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=OUTSIDE_CROP_LOGIT,
        )
