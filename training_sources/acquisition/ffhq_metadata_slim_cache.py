"""Slim, memory-cheap cache of the FFHQ metadata fields this pipeline uses.

`ffhq-dataset-v2.json` is 255 MB, and `json.load` on it peaks near 2 GB (per-image landmark lists
become millions of Python objects). This module parses it ONE entry at a time with
`json.JSONDecoder.raw_decode` (peak about the file size), keeps only the fields we use, and writes
one JSONL row per image to `FFHQ_SLIM_METADATA_JSONL_PATH`. Every later stage reads the slim file.

Run once (it is skipped when the cache exists): `python -m training_sources.acquisition.ffhq_metadata_slim_cache`
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from training_sources.training_sources_paths import FFHQ_METADATA_JSON_PATH, FFHQ_SLIM_METADATA_JSONL_PATH

logger = logging.getLogger(__name__)

JSON_WHITESPACE_AND_SEPARATORS = " \t\r\n,:"


def iterate_top_level_json_object_items(json_text: str) -> Iterator[tuple[str, dict]]:
    """Yield (key, value) pairs of a top-level JSON object, decoding one value at a time."""
    json_decoder = json.JSONDecoder()
    position = json_text.index("{") + 1
    while True:
        while json_text[position] in JSON_WHITESPACE_AND_SEPARATORS:
            position += 1
        if json_text[position] == "}":
            return
        key, position = json_decoder.raw_decode(json_text, position)
        while json_text[position] in JSON_WHITESPACE_AND_SEPARATORS:
            position += 1
        value, position = json_decoder.raw_decode(json_text, position)
        yield key, value


def dlib_eye_centers_from_ffhq_landmarks(face_landmarks: list[list[float]]) -> tuple[list[float], list[float]]:
    """Return (image-left, image-right) eye centers from FFHQ's 68-point dlib landmarks (points 36-47).

    FFHQ aligned every face with these landmarks, so they are a reliable cross-check for YuNet.
    """
    landmark_array = np.asarray(face_landmarks, dtype=np.float64)
    eye_centers = sorted((landmark_array[36:42].mean(axis=0), landmark_array[42:48].mean(axis=0)), key=lambda eye_xy: eye_xy[0])
    return [round(float(value), 2) for value in eye_centers[0]], [round(float(value), 2) for value in eye_centers[1]]


def slim_ffhq_metadata_entry(ffhq_index: int, ffhq_metadata_entry: dict) -> dict:
    """Keep only the fields the pipeline uses from one full metadata entry."""
    photo_metadata = ffhq_metadata_entry["metadata"]
    image_metadata = ffhq_metadata_entry["image"]
    dlib_image_left_eye_xy, dlib_image_right_eye_xy = dlib_eye_centers_from_ffhq_landmarks(image_metadata["face_landmarks"])
    return {
        "ffhq_index": ffhq_index,
        "license": photo_metadata["license"],
        "license_url": photo_metadata["license_url"],
        "author": photo_metadata["author"],
        "photo_url": photo_metadata["photo_url"],
        "photo_title": photo_metadata["photo_title"],
        "file_md5": image_metadata["file_md5"],
        "file_size": image_metadata["file_size"],
        "dlib_image_left_eye_xy": dlib_image_left_eye_xy,
        "dlib_image_right_eye_xy": dlib_image_right_eye_xy,
    }


def build_slim_ffhq_metadata_cache(full_metadata_path: Path, slim_cache_path: Path) -> int:
    """Stream the full metadata JSON into the slim JSONL cache; return the number of rows written."""
    json_text = full_metadata_path.read_text()
    partial_cache_path = slim_cache_path.with_suffix(".jsonl.part")
    row_count = 0
    with open(partial_cache_path, "w") as cache_file:
        for key, ffhq_metadata_entry in iterate_top_level_json_object_items(json_text):
            cache_file.write(json.dumps(slim_ffhq_metadata_entry(int(key), ffhq_metadata_entry)) + "\n")
            row_count += 1
    partial_cache_path.replace(slim_cache_path)
    return row_count


def load_slim_ffhq_metadata(slim_cache_path: Path = FFHQ_SLIM_METADATA_JSONL_PATH) -> dict[int, dict]:
    """Return {ffhq_index: slim row}, building the cache first if it does not exist yet."""
    if not slim_cache_path.exists():
        logger.info("building slim FFHQ metadata cache at %s", slim_cache_path)
        build_slim_ffhq_metadata_cache(FFHQ_METADATA_JSON_PATH, slim_cache_path)
    with open(slim_cache_path) as cache_file:
        return {row["ffhq_index"]: row for row in map(json.loads, cache_file)}


def main() -> None:
    """Build the slim cache if missing and report its size."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logger.info("slim cache holds %d rows", len(load_slim_ffhq_metadata()))


if __name__ == "__main__":
    main()
