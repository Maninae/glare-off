"""Filesystem locations and remote URLs shared by every training_sources stage.

Everything heavy lives on the external `vega` drive, never the internal disk or the repo:

- `SOURCES_ROOT` holds the FFHQ metadata, downloaded photos, lens masks, stage outputs, and review sheets.
- `LENS_SEGMENTATION_WEIGHTS_PATH` is the third-party lens segmenter, used offline as a labeling aid only.
"""

from pathlib import Path

SOURCES_ROOT = Path("/Volumes/vega/datasets/glare-off/sources")

FFHQ_METADATA_DIRECTORY = SOURCES_ROOT / "ffhq_metadata"
FFHQ_METADATA_JSON_PATH = FFHQ_METADATA_DIRECTORY / "ffhq-dataset-v2.json"
# One small JSONL row per image with only the fields we use (see ffhq_metadata_slim_cache.py).
FFHQ_SLIM_METADATA_JSONL_PATH = FFHQ_METADATA_DIRECTORY / "ffhq_metadata__slim-fields__n=70000.jsonl"
# DCGM/ffhq-features-dataset: one Azure Face API attribute JSON per image (glasses type).
DCGM_FEATURES_JSON_DIRECTORY = FFHQ_METADATA_DIRECTORY / "dcgm" / "json"
# royorel/FFHQ-Aging-Dataset: one CSV row per image with a Face++ glasses type.
FFHQ_AGING_LABELS_CSV_PATH = FFHQ_METADATA_DIRECTORY / "ffhq_aging_labels.csv"

FFHQ_PHOTO_DIRECTORY = SOURCES_ROOT / "ffhq_photos_1024"
LENS_MASK_DIRECTORY = SOURCES_ROOT / "lens_masks"
STAGE_OUTPUT_DIRECTORY = SOURCES_ROOT / "stages"
CANDIDATES_JSONL_PATH = STAGE_OUTPUT_DIRECTORY / "ffhq_candidates__license-and-glasses-filtered.jsonl"
DOWNLOAD_LOG_JSONL_PATH = STAGE_OUTPUT_DIRECTORY / "ffhq_downloads__md5-verified.jsonl"
ANNOTATIONS_JSONL_PATH = STAGE_OUTPUT_DIRECTORY / "ffhq_annotations__eyes-and-lens-masks.jsonl"
GLARE_SCORES_JSONL_PATH = STAGE_OUTPUT_DIRECTORY / "ffhq_glare_scores__per-face.jsonl"
SOURCE_MANIFEST_PATH = SOURCES_ROOT / "source_manifest.jsonl"
REVIEW_SHEET_DIRECTORY = SOURCES_ROOT / "review"
ATTRIBUTION_MARKDOWN_PATH = SOURCES_ROOT / "ATTRIBUTION.md"

# Byte-identical mirror of the official 1024x1024 PNGs (verified per file against the official md5).
FFHQ_HUGGING_FACE_MIRROR_URL = "https://huggingface.co/datasets/marcosv/ffhq-dataset/resolve/main"

LENS_SEGMENTATION_WEIGHTS_PATH = Path(
    "/Volumes/vega/ai-models/glare-off/lens-segmentation/segmentation_lenses_lraspp_mobilenet_v3_large.pth"
)
LENS_SEGMENTATION_WEIGHTS_URL = (
    "https://github.com/mantasu/glasses-detector/releases/download/v1.0.0/"
    "segmentation_lenses_lraspp_mobilenet_v3_large.pth"
)


def ffhq_photo_path(ffhq_index: int) -> Path:
    """Return where the 1024x1024 PNG of FFHQ image `ffhq_index` is stored locally."""
    return FFHQ_PHOTO_DIRECTORY / f"{ffhq_index:05d}.png"


def lens_mask_path(source_id: str) -> Path:
    """Return where the lens-mask PNG for `source_id` is stored."""
    return LENS_MASK_DIRECTORY / f"{source_id}__lens_mask.png"


def ffhq_mirror_url(ffhq_index: int) -> str:
    """Return the mirror URL of one FFHQ image (the mirror shards 10,000 images per `PartN` folder)."""
    part_number = ffhq_index // 10000 + 1
    return f"{FFHQ_HUGGING_FACE_MIRROR_URL}/Part{part_number}/{ffhq_index:05d}.png"
