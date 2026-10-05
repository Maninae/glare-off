"""Stage 1: pick the FFHQ images to download (license-kept clear-glasses faces plus no-glasses negatives).

Run: `python -m training_sources.acquisition.select_ffhq_candidates`

Reads the slim FFHQ metadata cache (`ffhq_metadata_slim_cache.py`, never the 2 GB-in-RAM full JSON) and the two glasses label sets, applies the license filter, and
writes one JSONL row per candidate to `CANDIDATES_JSONL_PATH` plus a counts JSON next to it.
Cheap (about a minute) and deterministic, so it simply rewrites its output on every run.
"""

import argparse
import json
import logging
from collections import Counter

from training_sources.acquisition.ffhq_glasses_labels import GlassesCategory, load_combined_glasses_categories
from training_sources.acquisition.ffhq_license_filter import classify_ffhq_license
from training_sources.acquisition.ffhq_metadata_slim_cache import load_slim_ffhq_metadata
from training_sources.source_split_assignment import hash_fraction_of_grouping_key
from training_sources.training_sources_paths import (
    CANDIDATES_JSONL_PATH,
    DCGM_FEATURES_JSON_DIRECTORY,
    FFHQ_AGING_LABELS_CSV_PATH,
    STAGE_OUTPUT_DIRECTORY,
)

logger = logging.getLogger(__name__)

FFHQ_IMAGE_COUNT = 70000
DEFAULT_NO_GLASSES_NEGATIVE_COUNT = 1000
CANDIDATE_COUNTS_JSON_PATH = STAGE_OUTPUT_DIRECTORY / "ffhq_candidates__stage-counts.json"


def ffhq_source_id(ffhq_index: int) -> str:
    """Return the manifest `source_id` of an FFHQ image."""
    return f"ffhq_{ffhq_index:05d}"


def build_candidate_row(ffhq_index: int, slim_metadata_row: dict, glasses_category: GlassesCategory) -> dict | None:
    """Return the candidate row for one FFHQ image, or None when its license is not kept."""
    source_license = classify_ffhq_license(slim_metadata_row["license_url"])
    if source_license is None:
        return None
    return {
        "source_id": ffhq_source_id(ffhq_index),
        "ffhq_index": ffhq_index,
        "has_glasses": glasses_category == GlassesCategory.CLEAR_GLASSES,
        "license": source_license.value,
        "author": slim_metadata_row["author"],
        "photo_url": slim_metadata_row["photo_url"],
        "photo_title": slim_metadata_row["photo_title"],
        "file_md5": slim_metadata_row["file_md5"],
        "file_size": slim_metadata_row["file_size"],
        "dlib_image_left_eye_xy": slim_metadata_row["dlib_image_left_eye_xy"],
        "dlib_image_right_eye_xy": slim_metadata_row["dlib_image_right_eye_xy"],
    }


def select_ffhq_candidates(no_glasses_negative_count: int) -> tuple[list[dict], dict]:
    """Return (candidate rows, stage counts) for clear-glasses faces and no-glasses negatives.

    - Negatives are the license-kept NO_GLASSES images with the smallest source-id hash, so the
      choice is deterministic and does not depend on dataset order.
    """
    slim_ffhq_metadata = load_slim_ffhq_metadata()
    glasses_categories = load_combined_glasses_categories(
        DCGM_FEATURES_JSON_DIRECTORY, FFHQ_AGING_LABELS_CSV_PATH, FFHQ_IMAGE_COUNT
    )
    glasses_rows, no_glasses_rows = [], []
    license_counter_by_category: dict[str, Counter] = {category.value: Counter() for category in GlassesCategory}
    for ffhq_index in range(FFHQ_IMAGE_COUNT):
        glasses_category = glasses_categories[ffhq_index]
        candidate_row = build_candidate_row(ffhq_index, slim_ffhq_metadata[ffhq_index], glasses_category)
        license_name = candidate_row["license"] if candidate_row else "dropped:" + slim_ffhq_metadata[ffhq_index]["license"]
        license_counter_by_category[glasses_category.value][license_name] += 1
        if candidate_row is None:
            continue
        if glasses_category == GlassesCategory.CLEAR_GLASSES:
            glasses_rows.append(candidate_row)
        elif glasses_category == GlassesCategory.NO_GLASSES:
            no_glasses_rows.append(candidate_row)
    no_glasses_rows.sort(key=lambda row: hash_fraction_of_grouping_key(row["source_id"]))
    selected_no_glasses_rows = sorted(no_glasses_rows[:no_glasses_negative_count], key=lambda row: row["ffhq_index"])
    stage_counts = {
        "ffhq_total": FFHQ_IMAGE_COUNT,
        "labelled_clear_glasses": sum(license_counter_by_category[GlassesCategory.CLEAR_GLASSES.value].values()),
        "labelled_no_glasses": sum(license_counter_by_category[GlassesCategory.NO_GLASSES.value].values()),
        "clear_glasses_license_kept": len(glasses_rows),
        "no_glasses_license_kept": len(no_glasses_rows),
        "no_glasses_selected": len(selected_no_glasses_rows),
        "license_breakdown_by_glasses_category": {
            category: dict(counter.most_common()) for category, counter in license_counter_by_category.items()
        },
    }
    return glasses_rows + selected_no_glasses_rows, stage_counts


def main() -> None:
    """Write the candidate JSONL and the stage counts."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--no-glasses-negative-count", type=int, default=DEFAULT_NO_GLASSES_NEGATIVE_COUNT)
    arguments = argument_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    candidate_rows, stage_counts = select_ffhq_candidates(arguments.no_glasses_negative_count)
    STAGE_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_JSONL_PATH, "w") as candidates_file:
        for candidate_row in candidate_rows:
            candidates_file.write(json.dumps(candidate_row) + "\n")
    with open(CANDIDATE_COUNTS_JSON_PATH, "w") as counts_file:
        json.dump(stage_counts, counts_file, indent=2)
    logger.info("wrote %d candidates to %s", len(candidate_rows), CANDIDATES_JSONL_PATH)
    logger.info("stage counts: %s", json.dumps(stage_counts, indent=2))


if __name__ == "__main__":
    main()
