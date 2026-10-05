"""Stage 3: per downloaded face, find the eyes and segment the lenses (glare is scored in stage 4).

Run: `python -m training_sources.labeling.annotate_sources [--worker-count 2] [--limit N]`

For each candidate photo it appends one row to `ANNOTATIONS_JSONL_PATH` and, when the lenses pass,
writes the lens label PNG (0 background, 1 image-left lens, 2 image-right lens). Resumable: source
ids already present in the annotations file are skipped, so a killed run just restarts.

Row `status` values:
- `ok`: eyes found and (for glasses faces) a plausible lens mask; (no-glasses faces) segmenter agrees there are no lenses.
- `photo_missing`, `no_face`: photo not downloaded, or YuNet found no face near the image centre.
- `eyes_disagree_with_ffhq_landmarks`: YuNet's eyes are far from FFHQ's own dlib eye landmarks
  (seen on close-ups where YuNet puts both eyes inside one lens).
- `lens_mask_failed`: glasses face whose lens segmentation is implausible (reason in `lens_failure_reason`).
- `negative_has_lenses`: retired status from an earlier run (the segmenter is not trusted on bare
  faces); re-run those rows with `--redo-status negative_has_lenses`.
"""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from eye_crop.yunet_eye_detector import DetectedFaceEyes, detect_face_eyes_in_photo
from training_sources.acquisition.download_ffhq_images import load_candidate_rows
from training_sources.labeling.lens_mask_assignment import LensMaskStatus, assign_left_right_lenses
from training_sources.labeling.lens_segmentation_model import LensSegmenter
from training_sources.stage_jsonl_io import load_jsonl_rows
from training_sources.training_sources_paths import (
    ANNOTATIONS_JSONL_PATH,
    CANDIDATES_JSONL_PATH,
    DCGM_FEATURES_JSON_DIRECTORY,
    LENS_MASK_DIRECTORY,
    ffhq_photo_path,
    lens_mask_path,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = 2
# Machine memory budget: at most 2 inference processes, each well under 2 GB (CPU torch, 1 thread).
MAXIMUM_WORKER_COUNT = 2
TORCH_THREADS_PER_WORKER = 1
# FFHQ is face-aligned: the target face's eye midpoint lies near the image centre.
MAXIMUM_EYE_MIDPOINT_OFFSET_FROM_CENTRE_FRACTION = 0.2
# Head yaw (DCGM / Azure Face, degrees) beyond which a single visible lens is acceptable.
STRONG_HEAD_YAW_DEGREES = 30.0
# Largest allowed YuNet-vs-dlib eye center gap, in dlib eye distances (median gap is about 0.12).
MAXIMUM_EYE_DISAGREEMENT_IN_EYE_DISTANCES = 0.3
PROGRESS_LOG_INTERVAL = 200
REJECTED_RAW_SEGMENTATION_DIRECTORY = LENS_MASK_DIRECTORY / "rejected_raw_segmentation"

worker_lens_segmenter: LensSegmenter | None = None


def initialize_annotation_worker() -> None:
    """Load the lens network once per worker process and cap its CPU threads."""
    global worker_lens_segmenter
    torch.set_num_threads(TORCH_THREADS_PER_WORKER)
    cv2.setNumThreads(1)
    worker_lens_segmenter = LensSegmenter()


def select_central_face(detected_faces: list[DetectedFaceEyes], image_width: int, image_height: int) -> DetectedFaceEyes | None:
    """Return the highest-scoring face whose eye midpoint is near the image centre, or None."""
    image_centre = np.array([image_width / 2.0, image_height / 2.0])
    maximum_offset = MAXIMUM_EYE_MIDPOINT_OFFSET_FROM_CENTRE_FRACTION * max(image_width, image_height)
    for detected_face in detected_faces:  # already sorted by score, best first
        eye_midpoint = (detected_face.image_left_eye_xy + detected_face.image_right_eye_xy) / 2.0
        if np.linalg.norm(eye_midpoint - image_centre) <= maximum_offset:
            return detected_face
    return None


def measure_eye_disagreement_with_dlib(detected_face: DetectedFaceEyes, candidate_row: dict) -> float:
    """Return the larger YuNet-to-dlib eye center gap, in units of the dlib eye distance."""
    dlib_left_eye = np.asarray(candidate_row["dlib_image_left_eye_xy"], dtype=np.float64)
    dlib_right_eye = np.asarray(candidate_row["dlib_image_right_eye_xy"], dtype=np.float64)
    dlib_eye_distance = float(np.linalg.norm(dlib_right_eye - dlib_left_eye))
    larger_gap = max(
        np.linalg.norm(detected_face.image_left_eye_xy - dlib_left_eye),
        np.linalg.norm(detected_face.image_right_eye_xy - dlib_right_eye),
    )
    return float(larger_gap / dlib_eye_distance)


def load_dcgm_head_yaw_degrees(ffhq_index: int) -> float | None:
    """Return the Azure Face head yaw for an FFHQ image, or None when DCGM found no face."""
    with open(DCGM_FEATURES_JSON_DIRECTORY / f"{ffhq_index:05d}.json") as features_file:
        detected_faces = json.load(features_file)
    if not detected_faces:
        return None
    return float(detected_faces[0]["faceAttributes"]["headPose"]["yaw"])


def annotate_one_candidate(candidate_row: dict) -> dict:
    """Run eye detection and lens segmentation on one candidate and return its annotation row (never raises per item)."""
    source_id = candidate_row["source_id"]
    annotation_row = {"source_id": source_id, "has_glasses": candidate_row["has_glasses"]}
    photo_path = ffhq_photo_path(candidate_row["ffhq_index"])
    photo_bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR) if photo_path.exists() else None
    if photo_bgr is None:
        return annotation_row | {"status": "photo_missing"}
    photo_height, photo_width = photo_bgr.shape[:2]
    central_face = select_central_face(detect_face_eyes_in_photo(photo_bgr), photo_width, photo_height)
    if central_face is None:
        return annotation_row | {"status": "no_face"}
    head_yaw_degrees = load_dcgm_head_yaw_degrees(candidate_row["ffhq_index"])
    annotation_row |= {
        "photo_path": str(photo_path),
        "image_left_eye_xy": [round(float(value), 2) for value in central_face.image_left_eye_xy],
        "image_right_eye_xy": [round(float(value), 2) for value in central_face.image_right_eye_xy],
        "eye_distance_pixels": round(central_face.eye_distance_pixels, 2),
        "face_detection_score": round(central_face.detection_score, 4),
        "head_yaw_degrees": head_yaw_degrees,
        "eye_disagreement_with_dlib": round(measure_eye_disagreement_with_dlib(central_face, candidate_row), 4),
    }
    if annotation_row["eye_disagreement_with_dlib"] > MAXIMUM_EYE_DISAGREEMENT_IN_EYE_DISTANCES:
        return annotation_row | {"status": "eyes_disagree_with_ffhq_landmarks"}
    photo_rgb = np.ascontiguousarray(photo_bgr[:, :, ::-1])
    lens_logits = worker_lens_segmenter.predict_lens_logits_on_photo(
        photo_rgb, central_face.image_left_eye_xy, central_face.image_right_eye_xy
    )
    head_is_strongly_turned = head_yaw_degrees is not None and abs(head_yaw_degrees) >= STRONG_HEAD_YAW_DEGREES
    lens_assignment = assign_left_right_lenses(
        lens_logits > 0, central_face.image_left_eye_xy, central_face.image_right_eye_xy, head_is_strongly_turned
    )
    annotation_row |= {
        "lens_mask_status": lens_assignment.status.value,
        "lens_failure_reason": lens_assignment.failure_reason,
        "lens_area_in_eye_distance_squared": lens_assignment.lens_area_in_eye_distance_squared,
    }
    if not candidate_row["has_glasses"]:
        # Diagnostic only: the segmenter outlines bare eyes as "lenses" on most glasses-free faces
        # (checked on a review sheet), so the two labelers' agreement decides, not the segmenter.
        annotation_row["segmenter_found_two_lenses"] = lens_assignment.status == LensMaskStatus.TWO_LENSES
        cv2.imwrite(str(lens_mask_path(source_id)), np.zeros((photo_height, photo_width), dtype=np.uint8))
        return annotation_row | {"status": "ok", "lens_mask_path": str(lens_mask_path(source_id))}
    if lens_assignment.status == LensMaskStatus.FAILED:
        cv2.imwrite(str(REJECTED_RAW_SEGMENTATION_DIRECTORY / f"{source_id}__raw.png"), ((lens_logits > 0) * 255).astype(np.uint8))
        return annotation_row | {"status": "lens_mask_failed"}
    cv2.imwrite(str(lens_mask_path(source_id)), lens_assignment.lens_label_map)
    return annotation_row | {"status": "ok", "lens_mask_path": str(lens_mask_path(source_id))}


def load_annotated_source_ids(annotations_jsonl_path: Path, redo_statuses: set[str]) -> set[str]:
    """Return source ids already annotated, minus those whose newest row has a status to redo."""
    if not annotations_jsonl_path.exists():
        return set()
    latest_status_by_source_id = {row["source_id"]: row["status"] for row in load_jsonl_rows(annotations_jsonl_path)}
    return {source_id for source_id, status in latest_status_by_source_id.items() if status not in redo_statuses}


def annotate_pending_candidates(worker_count: int, limit: int | None, redo_statuses: set[str]) -> dict[str, int]:
    """Annotate every candidate not yet annotated (or whose latest status is in `redo_statuses`).

    Returns status counts for this run. A redone row is appended; readers keep the newest row per id.
    """
    LENS_MASK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REJECTED_RAW_SEGMENTATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    finished_source_ids = load_annotated_source_ids(ANNOTATIONS_JSONL_PATH, redo_statuses)
    pending_rows = [row for row in load_candidate_rows(CANDIDATES_JSONL_PATH) if row["source_id"] not in finished_source_ids]
    pending_rows = [row for row in pending_rows if ffhq_photo_path(row["ffhq_index"]).exists()][:limit]
    logger.info("%d already annotated, %d pending with a downloaded photo", len(finished_source_ids), len(pending_rows))
    status_counts: dict[str, int] = {}
    with Pool(worker_count, initializer=initialize_annotation_worker) as worker_pool, open(ANNOTATIONS_JSONL_PATH, "a") as annotations_file:
        for completed_count, annotation_row in enumerate(worker_pool.imap_unordered(annotate_one_candidate, pending_rows, chunksize=4), start=1):
            annotations_file.write(json.dumps(annotation_row) + "\n")
            annotations_file.flush()
            status_counts[annotation_row["status"]] = status_counts.get(annotation_row["status"], 0) + 1
            if completed_count % PROGRESS_LOG_INTERVAL == 0:
                logger.info("progress %d/%d %s", completed_count, len(pending_rows), status_counts)
    return status_counts


def main() -> None:
    """Annotate all downloaded, not-yet-annotated candidates."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    argument_parser.add_argument("--limit", type=int, default=None)
    argument_parser.add_argument("--redo-status", action="append", default=[], help="re-run rows whose latest status is this (repeatable)")
    arguments = argument_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    status_counts = annotate_pending_candidates(min(arguments.worker_count, MAXIMUM_WORKER_COUNT), arguments.limit, set(arguments.redo_status))
    logger.info("done: %s", status_counts)


if __name__ == "__main__":
    main()
