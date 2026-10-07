"""Entry point: `python -m glasses_classifier.build_eye_crop_cache [--config ...] [KEY=VALUE ...]`.

Cuts every manifest face into the app's 256x512 eye crop once and stores the crops as uint8
`.npy` memmaps on vega, so training epochs read pixels instead of decoding 1024 px PNGs.

Per split, two files under `CACHE.DIRECTORY`:
- `<split>__eye_crops__variants=K__256x512.npy`: uint8 (N, K, 256, 512, 3) RGB.
- `<split>__rows.jsonl`: one row per face, in the same order (`source_id`, `has_glasses`, `photo_path`).

Variants (train only; evaluation splits get K=1, the exact manifest eye centers, native scale):
- 0 .. K-2: eye centers jittered like glare_model's dataset (YuNet noise), native scale.
- K-1: jittered AND cut through glare_model's phone-capture simulation (scale 0.15-0.6, noise, JPEG),
  so the gate also sees the app's area-pre-shrink path and camera degradation.
Crops always come from `extract_eye_crop_with_area_prefilter`, exactly as in the app.
"""

import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, extract_eye_crop_with_area_prefilter
from glare_model.data.crop_augmentation import EyeCenterJitterSettings, jitter_eye_centers
from glare_model.data.phone_capture_simulation import (
    PhoneCaptureSimulationSettings,
    apply_capture_degradation,
    enlarge_crop_region_to_target_scale,
)
from glasses_classifier.config_loading import DEFAULT_CONFIG_PATH, load_glasses_classifier_config
from training_sources.load_source_manifest import SourceManifestRow, load_source_manifest

logger = logging.getLogger(__name__)

CROP_HEIGHT = 256
CROP_WIDTH = 512
TRAIN_SPLIT = "train"


def cache_crops_path(cache_directory: Path, split: str, variant_count: int) -> Path:
    """Path of the uint8 crop memmap for `split`."""
    return cache_directory / f"{split}__eye_crops__variants={variant_count}__{CROP_HEIGHT}x{CROP_WIDTH}.npy"


def cache_rows_path(cache_directory: Path, split: str) -> Path:
    """Path of the per-face JSONL index for `split`."""
    return cache_directory / f"{split}__rows.jsonl"


def config_node_to_lowercase_kwargs(config_node: DictConfig) -> dict:
    """UPPER_CASE config keys -> lowercase dataclass kwargs, lists -> tuples (picklable plain types)."""
    plain_values = OmegaConf.to_container(config_node, resolve=True)
    return {key.lower(): tuple(value) if isinstance(value, list) else value for key, value in plain_values.items()}


def cut_phone_capture_crop(
    photo_rgb_uint8: np.ndarray,
    left_eye_xy: np.ndarray,
    right_eye_xy: np.ndarray,
    capture_settings: PhoneCaptureSimulationSettings,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Enlarge the crop's photo region to a phone-like scale, degrade it, and cut the crop (uint8)."""
    photo_rgb_float = photo_rgb_uint8.astype(np.float32) / 255.0
    # The shared helper also resamples lens-label planes; the classifier has no labels, so pass empty ones.
    dummy_lens_planes = np.zeros((*photo_rgb_uint8.shape[:2], 3), dtype=np.float32)
    target_scale = random_generator.uniform(*capture_settings.crop_scale_range)
    photo_region, _, region_left_eye, region_right_eye = enlarge_crop_region_to_target_scale(
        photo_rgb_float, dummy_lens_planes, left_eye_xy, right_eye_xy, CROP_WIDTH, CROP_HEIGHT, target_scale
    )
    photo_region = apply_capture_degradation(photo_region, capture_settings, random_generator)
    affine = compute_photo_to_eye_crop_affine(region_left_eye, region_right_eye, CROP_WIDTH, CROP_HEIGHT)
    return extract_eye_crop_with_area_prefilter(photo_region, affine, CROP_WIDTH, CROP_HEIGHT)


def cut_crop_variants_for_row(task: tuple[SourceManifestRow, int, int, int, dict, dict]) -> np.ndarray:
    """Worker: return the (variant_count, 256, 512, 3) uint8 crops for one face (see module docstring)."""
    manifest_row, row_index, variant_count, cache_seed, jitter_kwargs, capture_kwargs = task
    cv2.setNumThreads(1)
    photo_rgb = manifest_row.load_photo_rgb()
    random_generator = np.random.default_rng([cache_seed, row_index])
    jitter_settings = EyeCenterJitterSettings(**jitter_kwargs)
    capture_settings = PhoneCaptureSimulationSettings(**capture_kwargs)
    crops = np.empty((variant_count, CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
    if variant_count == 1:
        affine = compute_photo_to_eye_crop_affine(manifest_row.image_left_eye_xy, manifest_row.image_right_eye_xy, CROP_WIDTH, CROP_HEIGHT)
        crops[0] = extract_eye_crop_with_area_prefilter(photo_rgb, affine, CROP_WIDTH, CROP_HEIGHT)
        return crops
    for variant_index in range(variant_count):
        left_eye, right_eye = jitter_eye_centers(manifest_row.image_left_eye_xy, manifest_row.image_right_eye_xy, jitter_settings, random_generator)
        if variant_index == variant_count - 1:
            crops[variant_index] = cut_phone_capture_crop(photo_rgb, left_eye, right_eye, capture_settings, random_generator)
        else:
            affine = compute_photo_to_eye_crop_affine(left_eye, right_eye, CROP_WIDTH, CROP_HEIGHT)
            crops[variant_index] = extract_eye_crop_with_area_prefilter(photo_rgb, affine, CROP_WIDTH, CROP_HEIGHT)
    return crops


def build_split_cache(manifest_rows: list[SourceManifestRow], split: str, config: DictConfig) -> Path:
    """Write the crop memmap and the row index for one split; return the memmap path (skips if complete)."""
    cache_directory = Path(config.CACHE.DIRECTORY)
    cache_directory.mkdir(parents=True, exist_ok=True)
    variant_count = int(config.CACHE.TRAIN_VARIANTS) if split == TRAIN_SPLIT else 1
    crops_path = cache_crops_path(cache_directory, split, variant_count)
    rows_path = cache_rows_path(cache_directory, split)
    if crops_path.exists() and rows_path.exists():
        logger.info("cache for %s exists, skipping: %s", split, crops_path)
        return crops_path

    partial_path = crops_path.with_suffix(".partial.npy")
    crop_memmap = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.uint8, shape=(len(manifest_rows), variant_count, CROP_HEIGHT, CROP_WIDTH, 3)
    )
    jitter_kwargs = config_node_to_lowercase_kwargs(config.CACHE.EYE_JITTER)
    capture_kwargs = config_node_to_lowercase_kwargs(config.CACHE.PHONE_CAPTURE)
    tasks = [(row, index, variant_count, int(config.CACHE.SEED), jitter_kwargs, capture_kwargs) for index, row in enumerate(manifest_rows)]
    with Pool(int(config.CACHE.BUILD_PROCESSES)) as pool:
        for row_index, crops in enumerate(pool.imap(cut_crop_variants_for_row, tasks, chunksize=8)):
            crop_memmap[row_index] = crops
            if row_index % 500 == 0:
                logger.info("%s: %d / %d faces cropped", split, row_index, len(manifest_rows))
    crop_memmap.flush()
    del crop_memmap
    partial_path.rename(crops_path)
    with open(rows_path, "w") as rows_file:
        for row in manifest_rows:
            rows_file.write(json.dumps({"source_id": row.source_id, "has_glasses": row.has_glasses, "photo_path": str(row.photo_path)}) + "\n")
    logger.info("wrote %s (%.2f GB)", crops_path, crops_path.stat().st_size / 1e9)
    return crops_path


def main() -> None:
    """Build the cache for every split named in `CACHE.SPLITS`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Cache the app-exact eye crops of every manifest face on vega.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("overrides", nargs="*", help="KEY=VALUE dotlist overrides")
    arguments = parser.parse_args()
    config = load_glasses_classifier_config(arguments.config, arguments.overrides)
    all_rows = load_source_manifest()
    for split in config.CACHE.SPLITS:
        build_split_cache([row for row in all_rows if row.split == split], split, config)


if __name__ == "__main__":
    main()
