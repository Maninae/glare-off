"""Typed loader for the source manifest, for `glare_synthesis/`, `glare_model/`, and `evaluation/`.

Usage:
    from training_sources.load_source_manifest import load_source_manifest
    training_rows = load_source_manifest(splits={"train"})
    real_glare_rows = load_source_manifest(splits={"real_glare_eval"})

- Contract keys (CLAUDE.md "Source manifest") become typed dataclass fields; eye centers are numpy arrays.
- Any other key in a row is kept, untyped, in `extra_fields` (consumers tolerate unknown keys).
- A row missing a contract key raises KeyError naming the row, so a malformed manifest fails loud.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from training_sources.training_sources_paths import SOURCE_MANIFEST_PATH

CONTRACT_KEYS = (
    "source_id",
    "split",
    "photo_path",
    "license",
    "attribution",
    "image_left_eye_xy",
    "image_right_eye_xy",
    "lens_mask_path",
    "source_glare_score",
)


@dataclass
class SourceManifestRow:
    """One clean source face: where its photo and lens mask are, and how it may be used."""

    source_id: str
    split: str
    photo_path: Path
    license: str
    attribution: str
    image_left_eye_xy: np.ndarray
    image_right_eye_xy: np.ndarray
    lens_mask_path: Path
    source_glare_score: float
    has_glasses: bool = True
    extra_fields: dict = field(default_factory=dict)

    def load_photo_rgb(self) -> np.ndarray:
        """Read the photo as an (H, W, 3) uint8 RGB array."""
        photo_bgr = cv2.imread(str(self.photo_path), cv2.IMREAD_COLOR)
        if photo_bgr is None:
            raise FileNotFoundError(f"photo missing for {self.source_id}: {self.photo_path}")
        return np.ascontiguousarray(photo_bgr[:, :, ::-1])

    def load_lens_label_map(self) -> np.ndarray:
        """Read the lens mask as (H, W) uint8: 0 background, 1 image-left lens, 2 image-right lens."""
        lens_label_map = cv2.imread(str(self.lens_mask_path), cv2.IMREAD_GRAYSCALE)
        if lens_label_map is None:
            raise FileNotFoundError(f"lens mask missing for {self.source_id}: {self.lens_mask_path}")
        return lens_label_map


def parse_source_manifest_row(manifest_row: dict) -> SourceManifestRow:
    """Turn one decoded JSON row into a SourceManifestRow."""
    missing_keys = [key for key in CONTRACT_KEYS if key not in manifest_row]
    if missing_keys:
        raise KeyError(f"manifest row {manifest_row.get('source_id', '?')} lacks contract keys {missing_keys}")
    return SourceManifestRow(
        source_id=manifest_row["source_id"],
        split=manifest_row["split"],
        photo_path=Path(manifest_row["photo_path"]),
        license=manifest_row["license"],
        attribution=manifest_row["attribution"],
        image_left_eye_xy=np.asarray(manifest_row["image_left_eye_xy"], dtype=np.float64),
        image_right_eye_xy=np.asarray(manifest_row["image_right_eye_xy"], dtype=np.float64),
        lens_mask_path=Path(manifest_row["lens_mask_path"]),
        source_glare_score=float(manifest_row["source_glare_score"]),
        has_glasses=bool(manifest_row.get("has_glasses", True)),
        extra_fields={key: value for key, value in manifest_row.items() if key not in CONTRACT_KEYS and key != "has_glasses"},
    )


def load_source_manifest(manifest_path: Path = SOURCE_MANIFEST_PATH, splits: set[str] | None = None) -> list[SourceManifestRow]:
    """Load manifest rows, optionally keeping only the given splits (e.g. {"train"} or {"val", "test"})."""
    with open(manifest_path) as manifest_file:
        manifest_rows = [parse_source_manifest_row(json.loads(line)) for line in manifest_file if line.strip()]
    if splits is None:
        return manifest_rows
    return [row for row in manifest_rows if row.split in splits]
