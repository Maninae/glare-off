"""Pre-bake downloaded HDRIs into compact environment maps the DataLoader workers can sample fast.

    python -m glare_synthesis.bake_hdri_environment_maps

Per HDRI, writes `<id>.npz` with:
- `environment_rgb`: float16 (512, 1024, 3) linear RGB equirect, scaled so the solid-angle-weighted
  mean luminance, with the brightest 0.5% (sun, bare bulbs) clipped first, is 1 (exposure is set at render time relative to the face). Pre-blurred by about
  one pixel so a sub-pixel sun keeps its energy when the mirror mapping minifies it.
- `emitter_lon_lat`, `emitter_weights`: up to EMITTER_SAMPLE_COUNT bright directions (radians),
  weighted by (luminance x solid angle) ** 0.35 so windows and sky compete with the sun, used to aim the reflection at lamps, windows and the sun.
Equirect convention: column -> longitude in [-pi, pi), row -> latitude from +pi/2 (top) to -pi/2.
Also writes `hdri_baked_manifest.jsonl` (download manifest rows + baked path + peak luminance).
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from glare_synthesis.color_science import REC709_LUMINANCE_WEIGHTS
from glare_synthesis.download_polyhaven_hdris import DEFAULT_HDRI_DIRECTORY

logger = logging.getLogger(__name__)

BAKED_ENVIRONMENT_WIDTH = 1024
BAKED_ENVIRONMENT_HEIGHT = 512
ENERGY_PRESERVING_BLUR_SIGMA = 1.0
# float16 overflows at 65504; brighter (sun core) values are clipped just below it after the blur.
FLOAT16_SAFE_MAXIMUM = 60000.0
EMITTER_SAMPLE_COUNT = 256
DIFFUSE_MEAN_CLIP_QUANTILE = 0.995
# A direction counts as an emitter if it is this many times brighter than the mean radiance.
EMITTER_MINIMUM_RELATIVE_LUMINANCE = 4.0
EMITTER_TOP_FRACTION = 0.02
# Aim weights use power ** this exponent so extended windows and bright sky compete with the tiny sun.
EMITTER_POWER_FLATTENING_EXPONENT = 0.35


def equirect_latitude_column(height: int) -> np.ndarray:
    """Latitude (radians) of each equirect row center, shape (H, 1)."""
    return (np.pi / 2.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * np.pi)[:, None]


def bake_environment_map(hdr_path: Path, random_generator: np.random.Generator) -> dict:
    """Load one .hdr and return the baked arrays (see module docstring)."""
    environment_bgr = cv2.imread(str(hdr_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if environment_bgr is None:
        raise FileNotFoundError(f"cannot read HDR: {hdr_path}")
    environment_rgb = np.nan_to_num(cv2.cvtColor(environment_bgr, cv2.COLOR_BGR2RGB).astype(np.float32), nan=0.0, posinf=0.0)
    environment_rgb = np.maximum(environment_rgb, 0.0)
    if environment_rgb.shape[:2] != (BAKED_ENVIRONMENT_HEIGHT, BAKED_ENVIRONMENT_WIDTH):
        environment_rgb = cv2.resize(environment_rgb, (BAKED_ENVIRONMENT_WIDTH, BAKED_ENVIRONMENT_HEIGHT), interpolation=cv2.INTER_AREA)
    environment_rgb = cv2.GaussianBlur(environment_rgb, (0, 0), ENERGY_PRESERVING_BLUR_SIGMA)
    solid_angle_weight = np.cos(equirect_latitude_column(BAKED_ENVIRONMENT_HEIGHT))
    luminance = environment_rgb @ REC709_LUMINANCE_WEIGHTS
    # Normalize by the diffuse (sun-robust) mean: portraits are usually exposed for a face in shade or
    # under sky light, not in direct sun, so the sun must not darken the rest of the reflected scene.
    diffuse_luminance = np.minimum(luminance, float(np.quantile(luminance, DIFFUSE_MEAN_CLIP_QUANTILE)))
    mean_luminance = float((diffuse_luminance * solid_angle_weight).sum() / (solid_angle_weight.sum() * BAKED_ENVIRONMENT_WIDTH))
    environment_rgb /= max(mean_luminance, 1e-9)
    luminance /= max(mean_luminance, 1e-9)
    emitter_threshold = max(EMITTER_MINIMUM_RELATIVE_LUMINANCE, float(np.quantile(luminance, 1.0 - EMITTER_TOP_FRACTION)))
    emitter_rows, emitter_columns = np.nonzero(luminance >= emitter_threshold)
    emitter_power = ((luminance * solid_angle_weight)[emitter_rows, emitter_columns]) ** EMITTER_POWER_FLATTENING_EXPONENT
    if len(emitter_power) > 0:
        chosen = random_generator.choice(len(emitter_power), size=min(EMITTER_SAMPLE_COUNT, len(emitter_power)), replace=False, p=emitter_power / emitter_power.sum())
        emitter_rows, emitter_columns, emitter_power = emitter_rows[chosen], emitter_columns[chosen], emitter_power[chosen]
    emitter_longitude = (emitter_columns + 0.5) / BAKED_ENVIRONMENT_WIDTH * 2.0 * np.pi - np.pi
    emitter_latitude = np.pi / 2.0 - (emitter_rows + 0.5) / BAKED_ENVIRONMENT_HEIGHT * np.pi
    return {
        "environment_rgb": np.minimum(environment_rgb, FLOAT16_SAFE_MAXIMUM).astype(np.float16),
        "emitter_lon_lat": np.stack([emitter_longitude, emitter_latitude], axis=1).astype(np.float32),
        "emitter_weights": (emitter_power / max(float(emitter_power.sum()), 1e-9)).astype(np.float32),
        "peak_relative_luminance": float(luminance.max()),
    }


def main() -> None:
    """Bake every downloaded HDRI and write the baked manifest."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--hdri-dir", type=Path, default=DEFAULT_HDRI_DIRECTORY)
    arguments = argument_parser.parse_args()
    baked_directory = arguments.hdri_dir / "baked_1024x512"
    baked_directory.mkdir(parents=True, exist_ok=True)
    download_rows = [json.loads(line) for line in (arguments.hdri_dir / "hdri_download_manifest.jsonl").read_text().splitlines() if line.strip()]
    baked_rows = []
    for row_index, download_row in enumerate(download_rows):
        baked_path = baked_directory / f"{download_row['hdri_id']}.npz"
        baked_arrays = bake_environment_map(Path(download_row["hdr_path"]), np.random.default_rng(row_index))
        np.savez(baked_path, environment_rgb=baked_arrays["environment_rgb"], emitter_lon_lat=baked_arrays["emitter_lon_lat"],
                 emitter_weights=baked_arrays["emitter_weights"])
        baked_rows.append({**download_row, "baked_path": str(baked_path), "emitter_count": int(len(baked_arrays["emitter_weights"])),
                           "peak_relative_luminance": baked_arrays["peak_relative_luminance"]})
    baked_manifest_path = arguments.hdri_dir / "hdri_baked_manifest.jsonl"
    baked_manifest_path.write_text("".join(json.dumps(row) + "\n" for row in baked_rows))
    logger.info("baked %d HDRIs into %s", len(baked_rows), baked_directory)


if __name__ == "__main__":
    main()
