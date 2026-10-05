"""Glasses labels for FFHQ from two independent public attribute sets, combined by agreement.

- DCGM/ffhq-features-dataset: Azure Face API, `glasses` in {NoGlasses, ReadingGlasses, Sunglasses,
  SwimmingGoggles}; 529 images have no detected face (empty JSON list).
- FFHQ-Aging (royorel): Face++, `glasses` in {None, Normal, Dark}.

A face counts as wearing clear glasses only when BOTH say so (ReadingGlasses AND Normal), and as
glasses-free only when both say so (NoGlasses AND None). Requiring agreement trades about 10% of
recall for far fewer sunglasses or no-glasses faces slipping into the lens-mask stage. The labels
are used for filtering only and never shipped (both sets are CC BY-NC-SA).
"""

import csv
import json
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class GlassesCategory(str, Enum):
    """Combined glasses label of one FFHQ face."""

    CLEAR_GLASSES = "clear_glasses"
    NO_GLASSES = "no_glasses"
    DISAGREEING_OR_DARK = "disagreeing_or_dark"


DCGM_CLEAR_GLASSES_LABEL = "ReadingGlasses"
DCGM_NO_GLASSES_LABEL = "NoGlasses"
FFHQ_AGING_CLEAR_GLASSES_LABEL = "Normal"
FFHQ_AGING_NO_GLASSES_LABEL = "None"
MISSING_LABEL = "missing"


def combine_glasses_labels(dcgm_glasses_label: str, ffhq_aging_glasses_label: str) -> GlassesCategory:
    """Combine the two labelers' strings into one category (agreement required for a positive)."""
    if dcgm_glasses_label == DCGM_CLEAR_GLASSES_LABEL and ffhq_aging_glasses_label == FFHQ_AGING_CLEAR_GLASSES_LABEL:
        return GlassesCategory.CLEAR_GLASSES
    if dcgm_glasses_label == DCGM_NO_GLASSES_LABEL and ffhq_aging_glasses_label == FFHQ_AGING_NO_GLASSES_LABEL:
        return GlassesCategory.NO_GLASSES
    return GlassesCategory.DISAGREEING_OR_DARK


def load_dcgm_glasses_label(dcgm_features_json_directory: Path, ffhq_index: int) -> str:
    """Return the DCGM `glasses` string for one image, or MISSING_LABEL when no face was found."""
    with open(dcgm_features_json_directory / f"{ffhq_index:05d}.json") as features_file:
        detected_faces = json.load(features_file)
    if not detected_faces:
        return MISSING_LABEL
    return detected_faces[0]["faceAttributes"]["glasses"]


def load_ffhq_aging_glasses_labels(ffhq_aging_labels_csv_path: Path) -> dict[int, str]:
    """Return {ffhq_index: Face++ glasses string} for every row of the FFHQ-Aging label CSV."""
    with open(ffhq_aging_labels_csv_path, newline="") as labels_file:
        return {int(row["image_number"]): row["glasses"] for row in csv.DictReader(labels_file)}


def load_combined_glasses_categories(
    dcgm_features_json_directory: Path, ffhq_aging_labels_csv_path: Path, ffhq_image_count: int
) -> dict[int, GlassesCategory]:
    """Return {ffhq_index: GlassesCategory} for all FFHQ images (reads 70k small JSON files)."""
    ffhq_aging_labels = load_ffhq_aging_glasses_labels(ffhq_aging_labels_csv_path)
    combined_categories = {}
    for ffhq_index in range(ffhq_image_count):
        dcgm_label = load_dcgm_glasses_label(dcgm_features_json_directory, ffhq_index)
        aging_label = ffhq_aging_labels.get(ffhq_index, MISSING_LABEL)
        combined_categories[ffhq_index] = combine_glasses_labels(dcgm_label, aging_label)
    logger.info("loaded combined glasses labels for %d FFHQ images", len(combined_categories))
    return combined_categories
