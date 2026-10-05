"""Adapters from the REAL sibling modules (`training_sources/`, `glare_synthesis/`) to the dataset.

This is the only file that knows those modules exist. Both are reached through a
`"package.module:function"` path from the config (see `configs/glare_model/data/real.yaml`), resolved
when the provider / synthesizer is built, so the fake-data path never imports them and a rename
on their side is a one-line config change here.

- `ManifestSourceFaceProvider` reads rows from `training_sources.load_source_manifest` (contract:
  the source manifest keys in the repo CLAUDE.md), keeps one split, and loads photo + lens mask.
- `build_glare_synthesis_module_synthesizer` returns the real `synthesize(clean_eye_crop,
  lens_label_mask, rng) -> data_dict` function.
"""

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from glare_model.data.source_face import GlareSynthesizer, SourceFace
from glare_model.registry import GLARE_SYNTHESIZER_REGISTRY, SOURCE_FACE_PROVIDER_REGISTRY

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "/Volumes/vega/datasets/glare-off/sources/source_manifest.jsonl"
DEFAULT_MANIFEST_LOADER_PATH = "training_sources.load_source_manifest:load_source_manifest"
DEFAULT_SYNTHESIZE_FUNCTION_PATH = "glare_synthesis.render_lens_glare:render_lens_glare_sample"
UINT8_MAX = 255.0


def resolve_function_path(function_path: str) -> Callable[..., Any]:
    """Import `"package.module:function"` and return the function, with a clear error if absent."""
    module_name, separator, function_name = function_path.partition(":")
    if not separator:
        raise ValueError(f"function path {function_path!r} must look like 'package.module:function'")
    module = importlib.import_module(module_name)
    if not hasattr(module, function_name):
        raise AttributeError(f"{module_name} has no {function_name!r}; update the path in the data config")
    return getattr(module, function_name)


def read_manifest_field(manifest_row: Any, key: str) -> Any:
    """Read `key` from a manifest row that is either a dict or an object with attributes."""
    if isinstance(manifest_row, dict):
        return manifest_row[key]
    return getattr(manifest_row, key)


def read_optional_manifest_field(manifest_row: Any, key: str, default: Any) -> Any:
    """Like `read_manifest_field`, but return `default` when the row lacks `key`."""
    if isinstance(manifest_row, dict):
        return manifest_row.get(key, default)
    return getattr(manifest_row, key, default)


@SOURCE_FACE_PROVIDER_REGISTRY.register
class ManifestSourceFaceProvider:
    """Real provider: one split of the source manifest, loading photos and lens masks from disk.

    Args:
        split: "train", "val", or "test".
        manifest_path: the source manifest JSONL.
        manifest_loader_path: `module:function` returning the manifest rows for a path.
        max_source_glare_score: drop rows above this score (None trusts the loader's filtering).
    """

    def __init__(
        self,
        split: str,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        manifest_loader_path: str = DEFAULT_MANIFEST_LOADER_PATH,
        max_source_glare_score: float | None = None,
    ):
        load_source_manifest = resolve_function_path(manifest_loader_path)
        all_rows = list(load_source_manifest(Path(manifest_path)))
        self.manifest_rows = [
            row
            for row in all_rows
            if read_manifest_field(row, "split") == split
            and (max_source_glare_score is None or read_manifest_field(row, "source_glare_score") <= max_source_glare_score)
        ]
        if not self.manifest_rows:
            raise ValueError(f"no {split!r} rows in {manifest_path} (of {len(all_rows)} total)")
        logger.info("manifest %s: %d %s faces of %d rows", manifest_path, len(self.manifest_rows), split, len(all_rows))

    def __len__(self) -> int:
        """Number of faces in this split."""
        return len(self.manifest_rows)

    def load_source_face(self, source_index: int) -> SourceFace:
        """Load the photo (as float sRGB) and lens label mask for manifest row `source_index`."""
        manifest_row = self.manifest_rows[source_index]
        photo_path = read_manifest_field(manifest_row, "photo_path")
        photo_bgr = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        if photo_bgr is None:
            raise FileNotFoundError(f"cannot read photo {photo_path}")
        lens_mask_path = read_manifest_field(manifest_row, "lens_mask_path")
        lens_label_mask = cv2.imread(str(lens_mask_path), cv2.IMREAD_GRAYSCALE)
        if lens_label_mask is None:
            # A face without glasses may have no mask file; it is all background by definition.
            if read_optional_manifest_field(manifest_row, "has_glasses", True):
                raise FileNotFoundError(f"cannot read lens mask {lens_mask_path}")
            lens_label_mask = np.zeros(photo_bgr.shape[:2], dtype=np.uint8)
        if lens_label_mask.shape != photo_bgr.shape[:2]:
            raise ValueError(f"lens mask {lens_label_mask.shape} does not match photo {photo_bgr.shape[:2]}")
        return SourceFace(
            source_id=str(read_manifest_field(manifest_row, "source_id")),
            photo_rgb=cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / UINT8_MAX,
            image_left_eye_xy=np.asarray(read_manifest_field(manifest_row, "image_left_eye_xy"), dtype=np.float64),
            image_right_eye_xy=np.asarray(read_manifest_field(manifest_row, "image_right_eye_xy"), dtype=np.float64),
            lens_label_mask=lens_label_mask.astype(np.uint8),
        )


@GLARE_SYNTHESIZER_REGISTRY.register
def build_glare_synthesis_module_synthesizer(
    synthesize_function_path: str = DEFAULT_SYNTHESIZE_FUNCTION_PATH,
) -> GlareSynthesizer:
    """Return the real `glare_synthesis` function `synthesize(clean_eye_crop, lens_label_mask, rng)`."""
    return resolve_function_path(synthesize_function_path)
