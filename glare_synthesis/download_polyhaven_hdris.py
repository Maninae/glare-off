"""Download a category-balanced set of Poly Haven HDRIs (1k .hdr) for image-based lens reflections.

    python -m glare_synthesis.download_polyhaven_hdris

- License: Poly Haven assets are CC0 1.0 (https://polyhaven.com/license, checked 2026-10-05:
  "You can use our assets for any purpose, including commercial work."). Recorded per row.
- Polite and resumable: one request at a time with a pause, finished files (md5-verified) are skipped,
  partial downloads go to `.part` and are renamed only after the md5 matches.
- Writes `hdri_download_manifest.jsonl` (id, category, all categories, license, URL, md5, path).
"""

import argparse
import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HDRI_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/hdri/polyhaven")
POLYHAVEN_ASSET_LIST_URL = "https://api.polyhaven.com/assets?t=hdris"
POLYHAVEN_FILES_URL = "https://api.polyhaven.com/files/{asset_id}"
POLYHAVEN_LICENSE = "CC0-1.0"
POLYHAVEN_LICENSE_URL = "https://polyhaven.com/license"
HTTP_USER_AGENT = "glare-off-research/0.1 (open-source eyeglass glare removal; polite sequential downloader)"
REQUEST_PAUSE_SECONDS = 0.4
REQUEST_TIMEOUT_SECONDS = 60
DOWNLOAD_RESOLUTION = "1k"
# Primary category per asset, assigned in this priority order, and how many of each to take.
CATEGORY_QUOTAS = {
    "studio": 45,
    "night": 45,
    "indoor": 85,
    "outdoor": 85,
}


def fetch_json(url: str) -> dict:
    """GET a JSON document with the project user agent."""
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def assign_primary_category(asset_categories: list[str]) -> str | None:
    """First quota category (in priority order) that the asset belongs to, else None."""
    for category in CATEGORY_QUOTAS:
        if category in asset_categories:
            return category
    return None


def select_balanced_asset_ids(asset_index: dict, seed: int) -> list[tuple[str, str]]:
    """Pick (asset_id, primary_category) pairs filling each category quota, reproducibly."""
    random_generator = np.random.default_rng(seed)
    ids_by_category: dict[str, list[str]] = {category: [] for category in CATEGORY_QUOTAS}
    for asset_id in sorted(asset_index):
        primary_category = assign_primary_category(asset_index[asset_id]["categories"])
        if primary_category is not None:
            ids_by_category[primary_category].append(asset_id)
    selected_pairs = []
    for category, quota in CATEGORY_QUOTAS.items():
        candidate_ids = ids_by_category[category]
        chosen_indices = random_generator.permutation(len(candidate_ids))[:quota]
        selected_pairs.extend((candidate_ids[index], category) for index in sorted(chosen_indices))
    return selected_pairs


def file_md5(file_path: Path) -> str:
    """Hex md5 of a file."""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def download_verified_file(url: str, expected_md5: str, destination_path: Path) -> None:
    """Download to `.part`, check md5, then rename into place."""
    partial_path = destination_path.with_suffix(destination_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": HTTP_USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        partial_path.write_bytes(response.read())
    if file_md5(partial_path) != expected_md5:
        raise ValueError(f"md5 mismatch for {url}")
    partial_path.rename(destination_path)


def main() -> None:
    """Select, download and record the HDRI set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--output-dir", type=Path, default=DEFAULT_HDRI_DIRECTORY)
    argument_parser.add_argument("--seed", type=int, default=0)
    arguments = argument_parser.parse_args()
    hdr_directory = arguments.output_dir / f"hdr_{DOWNLOAD_RESOLUTION}"
    hdr_directory.mkdir(parents=True, exist_ok=True)
    asset_index = fetch_json(POLYHAVEN_ASSET_LIST_URL)
    selected_pairs = select_balanced_asset_ids(asset_index, arguments.seed)
    logger.info("selected %d of %d HDRIs", len(selected_pairs), len(asset_index))
    manifest_rows = []
    for pair_index, (asset_id, primary_category) in enumerate(selected_pairs):
        destination_path = hdr_directory / f"{asset_id}_{DOWNLOAD_RESOLUTION}.hdr"
        try:
            file_info = fetch_json(POLYHAVEN_FILES_URL.format(asset_id=asset_id))["hdri"][DOWNLOAD_RESOLUTION]["hdr"]
            time.sleep(REQUEST_PAUSE_SECONDS)
            if not (destination_path.exists() and file_md5(destination_path) == file_info["md5"]):
                download_verified_file(file_info["url"], file_info["md5"], destination_path)
                time.sleep(REQUEST_PAUSE_SECONDS)
        except (OSError, ValueError, KeyError) as download_error:
            logger.warning("skipping %s: %s", asset_id, download_error)
            continue
        manifest_rows.append({
            "hdri_id": asset_id,
            "category": primary_category,
            "polyhaven_categories": asset_index[asset_id]["categories"],
            "license": POLYHAVEN_LICENSE,
            "license_url": POLYHAVEN_LICENSE_URL,
            "source_url": f"https://polyhaven.com/a/{asset_id}",
            "file_url": file_info["url"],
            "md5": file_info["md5"],
            "hdr_path": str(destination_path),
        })
        if pair_index % 25 == 0:
            logger.info("%d/%d done", pair_index + 1, len(selected_pairs))
    manifest_path = arguments.output_dir / "hdri_download_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in manifest_rows))
    logger.info("wrote %d rows to %s", len(manifest_rows), manifest_path)


if __name__ == "__main__":
    main()
