"""Stage 2: download the candidate FFHQ PNGs from the Hugging Face mirror, verifying each file's md5.

Run: `python -m training_sources.acquisition.download_ffhq_images [--worker-count 4] [--limit N]`

- Resumable: a file already on disk with the official md5 is skipped; a partial `.part` file is
  discarded and re-fetched (files are ~1.5 MB, so byte-range resume is not worth the complexity).
- Polite: at most `--worker-count` concurrent requests, exponential backoff with jitter on errors
  and on HTTP 429/5xx.
- Every file is checked against `file_md5` from the official `ffhq-dataset-v2.json`, so a mirror
  that served a different or corrupt file is caught and the image is reported as failed.
"""

import argparse
import hashlib
import json
import logging
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from training_sources.training_sources_paths import (
    CANDIDATES_JSONL_PATH,
    DOWNLOAD_LOG_JSONL_PATH,
    FFHQ_PHOTO_DIRECTORY,
    ffhq_mirror_url,
    ffhq_photo_path,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKER_COUNT = 4
MAXIMUM_ATTEMPTS = 6
INITIAL_BACKOFF_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1 << 16
PROGRESS_LOG_INTERVAL = 200
USER_AGENT = "glare-off-dataset-builder/1.0 (research; bounded concurrency)"


def compute_file_md5(file_path: Path) -> str:
    """Return the hex md5 of a file, read in chunks."""
    file_hash = hashlib.md5()
    with open(file_path, "rb") as opened_file:
        for chunk in iter(lambda: opened_file.read(DOWNLOAD_CHUNK_BYTES), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def fetch_url_to_file(url: str, destination_path: Path) -> None:
    """Stream one URL to `destination_path` (raises on HTTP or network errors)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response, open(destination_path, "wb") as output_file:
        for chunk in iter(lambda: response.read(DOWNLOAD_CHUNK_BYTES), b""):
            output_file.write(chunk)


def download_one_verified_image(candidate_row: dict) -> dict:
    """Download one candidate unless already present and verified; return a log row with the outcome.

    Returns a dict with `source_id`, `status` in {already_present, downloaded, failed}, and on
    failure an `error` string. Never raises for per-item errors, so one bad file cannot stop the batch.
    """
    ffhq_index = candidate_row["ffhq_index"]
    final_path = ffhq_photo_path(ffhq_index)
    expected_md5 = candidate_row["file_md5"]
    if final_path.exists() and compute_file_md5(final_path) == expected_md5:
        return {"source_id": candidate_row["source_id"], "status": "already_present"}
    partial_path = final_path.with_suffix(".png.part")
    last_error = "no attempt made"
    for attempt_index in range(MAXIMUM_ATTEMPTS):
        try:
            fetch_url_to_file(ffhq_mirror_url(ffhq_index), partial_path)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as download_error:
            last_error = f"{type(download_error).__name__}: {download_error}"
            backoff_seconds = INITIAL_BACKOFF_SECONDS * 2**attempt_index * (1 + random.random())
            logger.warning("%s attempt %d failed (%s); retrying in %.1fs", candidate_row["source_id"], attempt_index + 1, last_error, backoff_seconds)
            time.sleep(backoff_seconds)
            continue
        downloaded_md5 = compute_file_md5(partial_path)
        if downloaded_md5 != expected_md5:
            last_error = f"md5 mismatch: got {downloaded_md5}, expected {expected_md5}"
            logger.warning("%s %s", candidate_row["source_id"], last_error)
            continue
        partial_path.replace(final_path)
        return {"source_id": candidate_row["source_id"], "status": "downloaded"}
    return {"source_id": candidate_row["source_id"], "status": "failed", "error": last_error}


def load_candidate_rows(candidates_jsonl_path: Path) -> list[dict]:
    """Read the stage-1 candidate rows."""
    with open(candidates_jsonl_path) as candidates_file:
        return [json.loads(line) for line in candidates_file if line.strip()]


def download_all_candidates(candidate_rows: list[dict], worker_count: int) -> dict[str, int]:
    """Download every candidate with bounded concurrency, append outcomes to the log, return status counts."""
    FFHQ_PHOTO_DIRECTORY.mkdir(parents=True, exist_ok=True)
    status_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor, open(DOWNLOAD_LOG_JSONL_PATH, "a") as log_file:
        pending_futures = [executor.submit(download_one_verified_image, row) for row in candidate_rows]
        for completed_count, future in enumerate(as_completed(pending_futures), start=1):
            outcome_row = future.result()
            status_counts[outcome_row["status"]] = status_counts.get(outcome_row["status"], 0) + 1
            if outcome_row["status"] != "already_present":
                log_file.write(json.dumps(outcome_row) + "\n")
                log_file.flush()
            if completed_count % PROGRESS_LOG_INTERVAL == 0:
                logger.info("progress %d/%d %s", completed_count, len(candidate_rows), status_counts)
    return status_counts


def main() -> None:
    """Download all stage-1 candidates."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    argument_parser.add_argument("--limit", type=int, default=None, help="only the first N candidates (smoke test)")
    arguments = argument_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    candidate_rows = load_candidate_rows(CANDIDATES_JSONL_PATH)[: arguments.limit]
    status_counts = download_all_candidates(candidate_rows, min(arguments.worker_count, DEFAULT_WORKER_COUNT))
    logger.info("done: %s", status_counts)


if __name__ == "__main__":
    main()
