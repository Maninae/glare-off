"""Stage 5: join candidates, annotations, and glare scores into the source manifest (the CLAUDE.md contract).

Run: `python -m training_sources.build_source_manifest`

Decisions made here:
- Glasses faces with `source_glare_score <= SOURCE_GLARE_SCORE_THRESHOLD` get train/val/test by a
  hash of their Flickr account (`source_split_assignment.py`).
- Glasses faces above `REAL_GLARE_EVAL_SCORE_THRESHOLD` get `split: "real_glare_eval"` (never trained
  on), unless their id is listed in the human-review rejection file (mask errors or no visible glare).
- Glasses faces between the two thresholds are ambiguous and dropped from both uses.
- No-glasses negatives get train/val/test the same way, `has_glasses: false`, an all-zero lens mask,
  and score 0.
Always rewrites the manifest and a stage-counts JSON from the stage files, so it is idempotent.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from training_sources.acquisition.ffhq_license_filter import SOURCE_LICENSE_TO_CANONICAL_URL, SourceLicense, format_attribution_text
from training_sources.stage_jsonl_io import latest_annotation_row_per_source, load_jsonl_rows
from training_sources.labeling.source_glare_score import REAL_GLARE_EVAL_SCORE_THRESHOLD, SOURCE_GLARE_SCORE_THRESHOLD
from training_sources.source_split_assignment import SourceSplit, assign_split_by_grouping_key, flickr_account_from_photo_url
from training_sources.training_sources_paths import (
    ANNOTATIONS_JSONL_PATH,
    CANDIDATES_JSONL_PATH,
    GLARE_SCORES_JSONL_PATH,
    REVIEW_SHEET_DIRECTORY,
    SOURCE_MANIFEST_PATH,
    STAGE_OUTPUT_DIRECTORY,
)

logger = logging.getLogger(__name__)

REAL_GLARE_EVAL_REVIEW_REJECTIONS_PATH = REVIEW_SHEET_DIRECTORY / "real_glare_eval__rejected-by-human-review.txt"
MANIFEST_COUNTS_JSON_PATH = STAGE_OUTPUT_DIRECTORY / "source_manifest__stage-counts.json"


def load_review_rejected_source_ids(rejections_path: Path) -> set[str]:
    """Return source ids listed (one per line, `#` comments allowed) in the review rejection file."""
    if not rejections_path.exists():
        return set()
    with open(rejections_path) as rejections_file:
        return {line.split("#")[0].strip() for line in rejections_file if line.split("#")[0].strip()}


def choose_manifest_split(candidate_row: dict, source_glare_score: float) -> SourceSplit | None:
    """Return the manifest split for an accepted face, or None when its glare score is ambiguous.

    Clean (or no-glasses) faces split by account hash; clearly glared faces go to the real-glare eval set.
    """
    if candidate_row["has_glasses"] and source_glare_score > REAL_GLARE_EVAL_SCORE_THRESHOLD:
        return SourceSplit.REAL_GLARE_EVAL
    if candidate_row["has_glasses"] and source_glare_score > SOURCE_GLARE_SCORE_THRESHOLD:
        return None
    return assign_split_by_grouping_key(flickr_account_from_photo_url(candidate_row["photo_url"]))


def build_manifest_row(candidate_row: dict, annotation_row: dict, source_glare_score: float, split: SourceSplit) -> dict:
    """Return one manifest row: the contract keys first, then extra keys consumers may ignore."""
    source_license = SourceLicense(candidate_row["license"])
    lens_count = {"two_lenses": 2, "single_lens": 1}.get(annotation_row.get("lens_mask_status", ""), 0)
    return {
        "source_id": candidate_row["source_id"],
        "split": split.value,
        "photo_path": annotation_row["photo_path"],
        "license": source_license.value,
        "attribution": format_attribution_text(source_license, candidate_row["author"], candidate_row["photo_url"], candidate_row["photo_title"]),
        "image_left_eye_xy": annotation_row["image_left_eye_xy"],
        "image_right_eye_xy": annotation_row["image_right_eye_xy"],
        "lens_mask_path": annotation_row["lens_mask_path"],
        "source_glare_score": source_glare_score,
        "has_glasses": candidate_row["has_glasses"],
        "lens_count": lens_count if candidate_row["has_glasses"] else 0,
        "license_url": SOURCE_LICENSE_TO_CANONICAL_URL[source_license],
        "photo_url": candidate_row["photo_url"],
        "author": candidate_row["author"],
        "photo_title": candidate_row["photo_title"],
        "flickr_account": flickr_account_from_photo_url(candidate_row["photo_url"]),
        "eye_distance_pixels": annotation_row["eye_distance_pixels"],
        "head_yaw_degrees": annotation_row.get("head_yaw_degrees"),
    }


def build_source_manifest_rows() -> tuple[list[dict], dict]:
    """Return (manifest rows sorted by source id, stage counts)."""
    candidates_by_source_id = {row["source_id"]: row for row in load_jsonl_rows(CANDIDATES_JSONL_PATH)}
    annotation_rows = latest_annotation_row_per_source(load_jsonl_rows(ANNOTATIONS_JSONL_PATH))
    glare_scores_by_source_id = {row["source_id"]: row["source_glare_score"] for row in load_jsonl_rows(GLARE_SCORES_JSONL_PATH)}
    review_rejected_source_ids = load_review_rejected_source_ids(REAL_GLARE_EVAL_REVIEW_REJECTIONS_PATH)
    manifest_rows = []
    annotation_status_counts: dict[str, Counter] = {"glasses": Counter(), "no_glasses": Counter()}
    for annotation_row in annotation_rows:
        candidate_row = candidates_by_source_id[annotation_row["source_id"]]
        group_name = "glasses" if candidate_row["has_glasses"] else "no_glasses"
        annotation_status_counts[group_name][annotation_row["status"]] += 1
        if annotation_row["status"] != "ok":
            continue
        source_glare_score = glare_scores_by_source_id.get(annotation_row["source_id"], 0.0) if candidate_row["has_glasses"] else 0.0
        split = choose_manifest_split(candidate_row, source_glare_score)
        if split is None:
            annotation_status_counts[group_name]["ambiguous_glare_dropped"] += 1
            continue
        if split == SourceSplit.REAL_GLARE_EVAL and annotation_row["source_id"] in review_rejected_source_ids:
            annotation_status_counts[group_name]["real_glare_eval_rejected_by_review"] += 1
            continue
        manifest_rows.append(build_manifest_row(candidate_row, annotation_row, source_glare_score, split))
    manifest_rows.sort(key=lambda row: row["source_id"])
    stage_counts = {
        "candidates": Counter("glasses" if row["has_glasses"] else "no_glasses" for row in candidates_by_source_id.values()),
        "annotation_status": {group: dict(counter.most_common()) for group, counter in annotation_status_counts.items()},
        "manifest_split_by_group": Counter(f"{'glasses' if row['has_glasses'] else 'no_glasses'}:{row['split']}" for row in manifest_rows),
        "manifest_license": Counter(row["license"] for row in manifest_rows),
        "manifest_license_trainable_glasses": Counter(
            row["license"] for row in manifest_rows if row["has_glasses"] and row["split"] != SourceSplit.REAL_GLARE_EVAL.value
        ),
        "manifest_lens_count_glasses": Counter(row["lens_count"] for row in manifest_rows if row["has_glasses"]),
        "glare_score_training_threshold": SOURCE_GLARE_SCORE_THRESHOLD,
        "glare_score_real_glare_eval_threshold": REAL_GLARE_EVAL_SCORE_THRESHOLD,
    }
    return manifest_rows, stage_counts


def main() -> None:
    """Rewrite the source manifest and its counts."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    manifest_rows, stage_counts = build_source_manifest_rows()
    with open(SOURCE_MANIFEST_PATH, "w") as manifest_file:
        for manifest_row in manifest_rows:
            manifest_file.write(json.dumps(manifest_row) + "\n")
    with open(MANIFEST_COUNTS_JSON_PATH, "w") as counts_file:
        json.dump(stage_counts, counts_file, indent=2, sort_keys=True)
    logger.info("wrote %d manifest rows to %s", len(manifest_rows), SOURCE_MANIFEST_PATH)
    logger.info("counts: %s", json.dumps(stage_counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
