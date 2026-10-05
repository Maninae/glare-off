"""Stage 4: compute `source_glare_score` for every accepted glasses face from its photo and lens mask.

Run: `python -m training_sources.labeling.score_source_glare [--worker-count 2]`

Kept apart from stage 3 because it is cheap (no network) and is the stage you re-run after changing
the scoring constants in `source_glare_score.py`. It always rewrites `GLARE_SCORES_JSONL_PATH` from
scratch: one row per `ok` glasses face with the score and per-lens features. No-glasses faces score 0.
"""

import argparse
import json
import logging
from multiprocessing import Pool

import cv2
import numpy as np

from training_sources.labeling.source_glare_score import compute_source_glare_score
from training_sources.stage_jsonl_io import latest_annotation_row_per_source, load_jsonl_rows
from training_sources.training_sources_paths import ANNOTATIONS_JSONL_PATH, GLARE_SCORES_JSONL_PATH

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = 2
MAXIMUM_WORKER_COUNT = 2
FEATURE_DECIMALS = 5


def score_one_annotated_face(annotation_row: dict) -> dict:
    """Return the glare-score row for one accepted glasses face."""
    cv2.setNumThreads(1)
    photo_bgr = cv2.imread(annotation_row["photo_path"], cv2.IMREAD_COLOR)
    lens_label_map = cv2.imread(annotation_row["lens_mask_path"], cv2.IMREAD_GRAYSCALE)
    if photo_bgr is None or lens_label_map is None:
        raise FileNotFoundError(f"photo or lens mask missing for {annotation_row['source_id']}")
    glare_score, per_lens_features = compute_source_glare_score(
        np.ascontiguousarray(photo_bgr[:, :, ::-1]),
        lens_label_map,
        np.asarray(annotation_row["image_left_eye_xy"]),
        np.asarray(annotation_row["image_right_eye_xy"]),
    )
    return {
        "source_id": annotation_row["source_id"],
        "source_glare_score": round(glare_score, FEATURE_DECIMALS),
        "glare_features_per_lens": [
            {name: round(value, FEATURE_DECIMALS) for name, value in vars(features).items()} for features in per_lens_features
        ],
    }


def score_all_accepted_glasses_faces(worker_count: int) -> int:
    """Score every `ok` glasses face and rewrite the scores file; return how many were scored."""
    annotation_rows = latest_annotation_row_per_source(load_jsonl_rows(ANNOTATIONS_JSONL_PATH))
    rows_to_score = [row for row in annotation_rows if row["status"] == "ok" and row["has_glasses"]]
    logger.info("scoring %d accepted glasses faces", len(rows_to_score))
    with Pool(worker_count) as worker_pool:
        score_rows = worker_pool.map(score_one_annotated_face, rows_to_score, chunksize=8)
    with open(GLARE_SCORES_JSONL_PATH, "w") as scores_file:
        for score_row in score_rows:
            scores_file.write(json.dumps(score_row) + "\n")
    return len(score_rows)


def main() -> None:
    """Rewrite the glare-score file for all accepted glasses faces."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    arguments = argument_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    scored_count = score_all_accepted_glasses_faces(min(arguments.worker_count, MAXIMUM_WORKER_COUNT))
    logger.info("wrote %d glare scores to %s", scored_count, GLARE_SCORES_JSONL_PATH)


if __name__ == "__main__":
    main()
