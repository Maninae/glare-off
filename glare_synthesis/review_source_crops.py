"""Clean crops + lens label masks for review sheets and benchmarks (not used in training).

Two sources:
- a directory of `<name>__clean_eye_crop.png` / `<name>__lens_label_mask.png` pairs (development crops);
- the source manifest, read with `training_sources.load_source_manifest` and cropped with `eye_crop.eye_crop_geometry`.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from eye_crop.eye_crop_geometry import EYE_CROP_HEIGHT, EYE_CROP_WIDTH, compute_photo_to_eye_crop_affine, extract_eye_crop
from training_sources.load_source_manifest import load_source_manifest

logger = logging.getLogger(__name__)

CLEAN_CROP_SUFFIX = "__clean_eye_crop.png"
LENS_MASK_SUFFIX = "__lens_label_mask.png"
DEFAULT_SOURCE_MANIFEST_PATH = Path("/Volumes/vega/datasets/glare-off/sources/source_manifest.jsonl")


@dataclass
class ReviewSourceCrop:
    """One clean crop ready for the renderer."""

    source_id: str
    clean_eye_crop: np.ndarray
    lens_label_mask: np.ndarray


def load_rgb_float_image(image_path: Path) -> np.ndarray:
    """Read an image file as float32 RGB in [0, 1]."""
    bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_crop_pairs_from_directory(pairs_directory: Path) -> list[ReviewSourceCrop]:
    """Load every crop/mask pair in a directory, sorted by name."""
    review_crops = []
    for clean_crop_path in sorted(pairs_directory.glob(f"*{CLEAN_CROP_SUFFIX}")):
        source_name = clean_crop_path.name.removesuffix(CLEAN_CROP_SUFFIX)
        lens_mask_path = pairs_directory / f"{source_name}{LENS_MASK_SUFFIX}"
        lens_label_mask = cv2.imread(str(lens_mask_path), cv2.IMREAD_GRAYSCALE)
        if lens_label_mask is None:
            raise FileNotFoundError(f"missing lens mask for {clean_crop_path.name}: {lens_mask_path}")
        review_crops.append(ReviewSourceCrop(source_name, load_rgb_float_image(clean_crop_path), lens_label_mask))
    logger.info("loaded %d development crop pairs from %s", len(review_crops), pairs_directory)
    return review_crops


def load_crops_from_source_manifest(manifest_path: Path, maximum_crop_count: int, random_generator: np.random.Generator,
                                    crop_width: int = EYE_CROP_WIDTH, crop_height: int = EYE_CROP_HEIGHT) -> list[ReviewSourceCrop]:
    """Crop a random subset of training-split faces with glasses to the aligned eye crop, with their lens masks."""
    manifest_rows = [row for row in load_source_manifest(manifest_path, splits={"train"}) if row.has_glasses]
    chosen_indices = random_generator.permutation(len(manifest_rows))[:maximum_crop_count]
    review_crops = []
    for row_index in chosen_indices:
        row = manifest_rows[int(row_index)]
        photo_rgb = row.load_photo_rgb().astype(np.float32) / 255.0
        photo_to_crop = compute_photo_to_eye_crop_affine(row.image_left_eye_xy, row.image_right_eye_xy, crop_width, crop_height)
        clean_eye_crop = np.clip(extract_eye_crop(photo_rgb, photo_to_crop, crop_width, crop_height), 0.0, 1.0)
        lens_label_mask = cv2.warpAffine(row.load_lens_label_map(), photo_to_crop, (crop_width, crop_height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        review_crops.append(ReviewSourceCrop(row.source_id, clean_eye_crop.astype(np.float32), lens_label_mask))
    logger.info("loaded %d manifest crops from %s", len(review_crops), manifest_path)
    return review_crops
