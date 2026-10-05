"""Access to the baked HDRI environment maps, with a small per-process cache for DataLoader workers.

`get_hdri_environment_library(path)` returns one library object per process (functools cache), and
the library keeps the last ENVIRONMENT_CACHE_SIZE maps as float32 (about 6 MB each), so a worker
holds well under 100 MB. Baking: `bake_hdri_environment_maps.py`.
"""

import functools
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_HDRI_BAKED_MANIFEST_PATH = Path("/Volumes/vega/datasets/glare-off/hdri/polyhaven/hdri_baked_manifest.jsonl")
ENVIRONMENT_CACHE_SIZE = 8


@dataclass
class HdriEnvironment:
    """One baked environment: mean-normalized linear RGB equirect plus bright-emitter directions."""

    hdri_id: str
    environment_rgb: np.ndarray
    emitter_lon_lat: np.ndarray
    emitter_weights: np.ndarray


class HdriEnvironmentLibrary:
    """Index of baked HDRIs by category, and an LRU cache of loaded maps."""

    def __init__(self, baked_manifest_path: Path):
        """Read the baked manifest; maps are loaded lazily."""
        if not baked_manifest_path.exists():
            raise FileNotFoundError(f"HDRI baked manifest missing: {baked_manifest_path} (run download_polyhaven_hdris then bake_hdri_environment_maps)")
        manifest_rows = [json.loads(line) for line in baked_manifest_path.read_text().splitlines() if line.strip()]
        self.baked_path_by_id = {row["hdri_id"]: Path(row["baked_path"]) for row in manifest_rows}
        self.ids_by_category: dict[str, list[str]] = {}
        for row in manifest_rows:
            self.ids_by_category.setdefault(row["category"], []).append(row["hdri_id"])
        self.loaded_environments: OrderedDict[str, HdriEnvironment] = OrderedDict()
        logger.info("HDRI library: %d maps in %s", len(manifest_rows), sorted(self.ids_by_category))

    def sample_hdri_id(self, random_generator: np.random.Generator, category_probabilities: dict[str, float]) -> str:
        """Pick a category (restricted to those present), then a uniform HDRI within it."""
        available_categories = [category for category in category_probabilities if self.ids_by_category.get(category)]
        weights = np.array([category_probabilities[category] for category in available_categories], dtype=np.float64)
        category = available_categories[random_generator.choice(len(available_categories), p=weights / weights.sum())]
        category_ids = self.ids_by_category[category]
        return category_ids[int(random_generator.integers(len(category_ids)))]

    def load_environment(self, hdri_id: str) -> HdriEnvironment:
        """Return the baked map for an id, from cache when possible."""
        if hdri_id in self.loaded_environments:
            self.loaded_environments.move_to_end(hdri_id)
            return self.loaded_environments[hdri_id]
        with np.load(self.baked_path_by_id[hdri_id]) as baked_arrays:
            environment = HdriEnvironment(
                hdri_id=hdri_id,
                environment_rgb=baked_arrays["environment_rgb"].astype(np.float32),
                emitter_lon_lat=baked_arrays["emitter_lon_lat"],
                emitter_weights=baked_arrays["emitter_weights"],
            )
        self.loaded_environments[hdri_id] = environment
        if len(self.loaded_environments) > ENVIRONMENT_CACHE_SIZE:
            self.loaded_environments.popitem(last=False)
        return environment


@functools.cache
def get_hdri_environment_library(baked_manifest_path: Path) -> HdriEnvironmentLibrary:
    """One library per process and manifest path."""
    return HdriEnvironmentLibrary(baked_manifest_path)
