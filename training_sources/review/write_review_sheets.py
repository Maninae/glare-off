"""Stage 6: write the human-review contact sheets to `REVIEW_SHEET_DIRECTORY`.

Run: `python -m training_sources.review.write_review_sheets`

Sheet sets (each split into grids of 40 tiles, 4 columns, 512x256 eye crops):
- `a_accepted-clean__random`: random accepted clean glasses faces (train/val/test).
- `b_glare-score__around-training-threshold`: faces just below and just above the training cut, by score.
- `c_real-glare-eval-candidates__by-score`: every face above the eval cut, mildest first, INCLUDING
  those already rejected by review (caption says so), so the rejection list can be audited.
- `d_rejected__*`: lens-mask failures (raw segmentation in red), eye-check failures, no face, and
  no-glasses negatives the segmenter thinks wear glasses.
- `e_no-glasses-negatives__random`: random accepted negatives.
Rewrites the sheet files each run; never touches the review rejection list.
"""

import logging
import random

from training_sources.build_source_manifest import REAL_GLARE_EVAL_REVIEW_REJECTIONS_PATH, load_review_rejected_source_ids
from training_sources.labeling.annotate_sources import REJECTED_RAW_SEGMENTATION_DIRECTORY
from training_sources.labeling.source_glare_score import REAL_GLARE_EVAL_SCORE_THRESHOLD, SOURCE_GLARE_SCORE_THRESHOLD
from training_sources.review.review_contact_sheets import TILES_PER_SHEET, load_review_tile_for_annotation, write_contact_sheets
from training_sources.stage_jsonl_io import latest_annotation_row_per_source, load_jsonl_rows
from training_sources.training_sources_paths import (
    ANNOTATIONS_JSONL_PATH,
    GLARE_SCORES_JSONL_PATH,
    REVIEW_SHEET_DIRECTORY,
    SOURCE_MANIFEST_PATH,
    ffhq_photo_path,
)

logger = logging.getLogger(__name__)

REVIEW_SAMPLE_SEED = 20261005
RANDOM_SAMPLE_SHEET_COUNT = 2
AROUND_THRESHOLD_TILES_EACH_SIDE = 40
REJECTED_TILES_PER_REASON = 40


def caption_for_row(row: dict) -> str:
    """Short tile caption: id, glare score, split or status."""
    score_text = f" g={row['source_glare_score']:.3f}" if "source_glare_score" in row else ""
    label_text = row.get("split") or row.get("status", "")
    reason_text = row.get("lens_failure_reason", "")
    return f"{row['source_id']}{score_text} {label_text} {reason_text}"[:58]


def render_tiles(rows: list[dict], include_raw_segmentation: bool = False) -> list:
    """Render a tile per row, skipping rows whose photo is missing."""
    tiles = []
    for row in rows:
        raw_segmentation_path = REJECTED_RAW_SEGMENTATION_DIRECTORY / f"{row['source_id']}__raw.png" if include_raw_segmentation else None
        tile = load_review_tile_for_annotation(row, caption_for_row(row), raw_segmentation_path)
        if tile is not None:
            tiles.append(tile)
    return tiles


def clear_previous_sheets() -> None:
    """Move old sheet JPEGs into `previous_sheets/` (overwriting older copies) so stale sheets never mix with new ones."""
    stale_directory = REVIEW_SHEET_DIRECTORY / "previous_sheets"
    stale_directory.mkdir(parents=True, exist_ok=True)
    for sheet_path in REVIEW_SHEET_DIRECTORY.glob("*.jpg"):
        sheet_path.replace(stale_directory / sheet_path.name)


def write_all_review_sheets() -> None:
    """Build every sheet set from the manifest and the annotation file."""
    random_generator = random.Random(REVIEW_SAMPLE_SEED)
    manifest_rows = load_jsonl_rows(SOURCE_MANIFEST_PATH)
    annotation_rows = latest_annotation_row_per_source(load_jsonl_rows(ANNOTATIONS_JSONL_PATH))
    clear_previous_sheets()
    sample_size = TILES_PER_SHEET * RANDOM_SAMPLE_SHEET_COUNT

    clean_glasses_rows = [row for row in manifest_rows if row["has_glasses"] and row["split"] in ("train", "val", "test")]
    clean_sample = random_generator.sample(clean_glasses_rows, min(sample_size, len(clean_glasses_rows)))
    write_contact_sheets(render_tiles(clean_sample), REVIEW_SHEET_DIRECTORY, "a_accepted-clean__random")

    # Scored faces come from annotations + scores, since ambiguous and review-rejected faces are not in the manifest.
    glare_scores_by_source_id = {row["source_id"]: row["source_glare_score"] for row in load_jsonl_rows(GLARE_SCORES_JSONL_PATH)}
    review_rejected_source_ids = load_review_rejected_source_ids(REAL_GLARE_EVAL_REVIEW_REJECTIONS_PATH)
    glasses_rows_by_score = sorted(
        (row | {"source_glare_score": glare_scores_by_source_id[row["source_id"]]} for row in annotation_rows if row["source_id"] in glare_scores_by_source_id),
        key=lambda row: row["source_glare_score"],
    )
    below_training_cut = [row for row in glasses_rows_by_score if row["source_glare_score"] <= SOURCE_GLARE_SCORE_THRESHOLD]
    above_training_cut = [row for row in glasses_rows_by_score if row["source_glare_score"] > SOURCE_GLARE_SCORE_THRESHOLD]
    around_training_cut = below_training_cut[-AROUND_THRESHOLD_TILES_EACH_SIDE:] + above_training_cut[:AROUND_THRESHOLD_TILES_EACH_SIDE]
    write_contact_sheets(render_tiles(around_training_cut), REVIEW_SHEET_DIRECTORY, "b_glare-score__around-training-threshold")

    eval_candidates = [
        row | {"status": "REVIEW-REJECTED" if row["source_id"] in review_rejected_source_ids else "eval"}
        for row in glasses_rows_by_score
        if row["source_glare_score"] > REAL_GLARE_EVAL_SCORE_THRESHOLD
    ]
    write_contact_sheets(render_tiles(eval_candidates), REVIEW_SHEET_DIRECTORY, "c_real-glare-eval-candidates__by-score")

    for rejection_status in ("lens_mask_failed", "eyes_disagree_with_ffhq_landmarks", "no_face"):
        rejected_rows = [row for row in annotation_rows if row["status"] == rejection_status]
        rejected_sample = random_generator.sample(rejected_rows, min(REJECTED_TILES_PER_REASON, len(rejected_rows)))
        rejected_sample = [row | {"photo_path": row.get("photo_path") or default_photo_path(row)} for row in rejected_sample]
        tiles = render_tiles(rejected_sample, include_raw_segmentation=True)
        if tiles:
            write_contact_sheets(tiles, REVIEW_SHEET_DIRECTORY, f"d_rejected__{rejection_status}")

    negative_rows = [row for row in manifest_rows if not row["has_glasses"]]
    negative_sample = random_generator.sample(negative_rows, min(TILES_PER_SHEET, len(negative_rows)))
    write_contact_sheets(render_tiles(negative_sample), REVIEW_SHEET_DIRECTORY, "e_no-glasses-negatives__random")
    logger.info("review rejection list (edit by hand, then rebuild the manifest): %s", REAL_GLARE_EVAL_REVIEW_REJECTIONS_PATH)


def default_photo_path(annotation_row: dict) -> str:
    """Photo path for rows that stopped before recording it (e.g. no face found)."""
    return str(ffhq_photo_path(int(annotation_row["source_id"].split("_")[1])))


def main() -> None:
    """Write every review sheet set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    write_all_review_sheets()


if __name__ == "__main__":
    main()
