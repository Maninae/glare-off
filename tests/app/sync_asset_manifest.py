"""Regenerate app/asset_manifest.json and stamp its version into app/sw.js.

The manifest lists every file the app serves with its byte size. Three readers:
- sw.js precaches the entries with `"precache": true` and versions its cache by the hash.
- main.js sums the sizes of the files a visitor actually downloads for a real progress bar
  (Content-Length lies when a host gzips).
- the size report in app/CLAUDE.md.

Run after adding, removing or changing any file in app/ (the browser test fails while stale):
    /Volumes/vega/datasets/glare-off/venv/bin/python -m tests.app.sync_asset_manifest
    /Volumes/vega/datasets/glare-off/venv/bin/python -m tests.app.sync_asset_manifest --check
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

APP_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "app"
MANIFEST_PATH = APP_DIRECTORY / "asset_manifest.json"
SERVICE_WORKER_PATH = APP_DIRECTORY / "sw.js"
SERVICE_WORKER_VERSION_PATTERN = re.compile(r'^const ASSET_MANIFEST_VERSION = "[0-9a-f]*";$', re.MULTILINE)

# Files in app/ that are never served to visitors (docs for developers, the stock detector the
# Python side uses, the stand-in model once the real one exists, the manifest itself).
NOT_SERVED_NAMES = {"CLAUDE.md", "asset_manifest.json", ".DS_Store", "README.md", "face_detection_yunet_2023mar.onnx"}
# Cached at runtime when first fetched (through the service worker, so never twice): the
# models and ORT build (the worker shows real progress), and the link preview (for unfurl bots).
NOT_PRECACHED_PREFIXES = ("vendor/", "models/", "icons/preview")


def list_served_files() -> list[Path]:
    """Every file under app/ a browser may request, sorted."""
    served_files = []
    for file_path in sorted(APP_DIRECTORY.rglob("*")):
        if not file_path.is_file() or file_path.name in NOT_SERVED_NAMES:
            continue
            continue
        served_files.append(file_path)
    return served_files


def build_manifest() -> dict:
    """Manifest dict with a content hash over every listed file."""
    content_hash = hashlib.sha256()
    assets = []
    for file_path in list_served_files():
        relative_path = file_path.relative_to(APP_DIRECTORY).as_posix()
        if relative_path == "sw.js":
            continue  # the service worker is fetched by the browser itself, never cached by itself
        file_bytes = file_path.read_bytes()
        content_hash.update(relative_path.encode() + b"\0" + hashlib.sha256(file_bytes).digest())
        assets.append({"path": relative_path, "bytes": len(file_bytes), "precache": not relative_path.startswith(NOT_PRECACHED_PREFIXES)})
    return {"version": content_hash.hexdigest()[:16], "assets": assets}


def stamp_service_worker_version(service_worker_source: str, version: str) -> str:
    """Rewrite the one version line in sw.js."""
    if not SERVICE_WORKER_VERSION_PATTERN.search(service_worker_source):
        raise ValueError("sw.js has no ASSET_MANIFEST_VERSION line to stamp")
    return SERVICE_WORKER_VERSION_PATTERN.sub(f'const ASSET_MANIFEST_VERSION = "{version}";', service_worker_source)


def main() -> None:
    """Write (or with --check, verify) the manifest and the sw.js version."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the manifest or sw.js is stale")
    arguments = parser.parse_args()
    manifest = build_manifest()
    manifest_text = json.dumps(manifest, indent=1) + "\n"
    service_worker_text = stamp_service_worker_version(SERVICE_WORKER_PATH.read_text(), manifest["version"])
    if arguments.check:
        stale = MANIFEST_PATH.read_text() != manifest_text or SERVICE_WORKER_PATH.read_text() != service_worker_text
        print("asset manifest is STALE; run tests.app.sync_asset_manifest" if stale else "asset manifest is current")
        sys.exit(1 if stale else 0)
    MANIFEST_PATH.write_text(manifest_text)
    SERVICE_WORKER_PATH.write_text(service_worker_text)
    precached_bytes = sum(asset["bytes"] for asset in manifest["assets"] if asset["precache"])
    print(f"wrote {len(manifest['assets'])} assets, version {manifest['version']}, precache {precached_bytes / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
