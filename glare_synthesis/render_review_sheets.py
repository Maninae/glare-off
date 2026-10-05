"""Regenerate the glare-synthesis review contact sheets (look at these before trusting the renderer).

    python -m glare_synthesis.render_review_sheets --output-dir /Volumes/vega/datasets/glare-off/synthesis-review/latest

Sheets written:
- `light_type__<type>.png`: 12 random single-source procedural samples per light type (4 x 3 grid).
- `hdri__<category>.png`: 12 image-based (mirrored HDRI) samples per HDRI category.
- `default_distribution.png`: 24 random samples from the default training distribution (what the model sees).
- `clean_glared_masks.png`: rows of clean | glared | glare_mask | lost_detail_mask, default distribution.
- `severity_ladder.png`: one fixed scene per crop, brightness rising left to right.
- `real_vs_synthetic.png` (only with --real-reference-dir): each real glare crop beside four HDRI look-alikes.
Sources: the source manifest if it exists (or --manifest-path), else --development-pairs-dir.
"""

import argparse
import dataclasses
import logging
from pathlib import Path

import cv2
import numpy as np

from glare_synthesis.glare_sampling_config import GlareSamplingConfig, GlareSeverity, LightSourceType
from glare_synthesis.glare_scene_sampling import sample_glare_scene
from glare_synthesis.lens_geometry import analyze_lens_label_mask
from glare_synthesis.render_lens_glare import hdri_library_for_config, render_glare_scene, render_lens_glare_sample
from glare_synthesis.review_source_crops import (
    DEFAULT_SOURCE_MANIFEST_PATH,
    ReviewSourceCrop,
    load_crop_pairs_from_directory,
    load_crops_from_source_manifest,
    load_rgb_float_image,
)

logger = logging.getLogger(__name__)

DEFAULT_REVIEW_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/synthesis-review/latest")
DEFAULT_DEVELOPMENT_PAIRS_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/synthesis-review/dev-crops/pairs")
SAMPLES_PER_LIGHT_TYPE = 12
LIGHT_TYPE_GRID_COLUMNS = 4
MASK_SHEET_ROW_COUNT = 10
SEVERITY_LADDER_SCALES = (0.03, 0.08, 0.2, 0.45, 1.0, 2.5, 6.0)
MANIFEST_REVIEW_CROP_COUNT = 24
LABEL_STRIP_HEIGHT = 18
LABEL_FONT_SCALE = 0.5
DEFAULT_DISTRIBUTION_SAMPLE_COUNT = 24
LOOKALIKES_PER_REFERENCE = 4
HDRI_CATEGORIES = ("indoor", "outdoor", "studio", "night")
# Real glare crop name -> (HDRI category that should reproduce its look, clean twin crop name).
REAL_REFERENCE_LOOKALIKES = {
    "ring-light-glare-glasses-before.png": ("studio", "ring-light"),
    "sunlight-glare-glasses-before.png": ("indoor", "sunlight"),
    "heavy-reflection-glasses-before.png": ("outdoor", "heavy-reflection"),
}


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """float [0, 1] (H, W, 3) or (H, W, 1) -> uint8 (H, W, 3)."""
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def labeled_tile(image: np.ndarray, label_text: str) -> np.ndarray:
    """Put a small dark label strip above a tile."""
    tile = to_uint8_rgb(image)
    label_strip = np.full((LABEL_STRIP_HEIGHT, tile.shape[1], 3), 24, np.uint8)
    cv2.putText(label_strip, label_text, (4, LABEL_STRIP_HEIGHT - 5), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([label_strip, tile])


def tile_grid(tiles: list[np.ndarray], column_count: int) -> np.ndarray:
    """Arrange equal-size uint8 tiles in rows of `column_count` (padding the last row with black)."""
    blank_tile = np.zeros_like(tiles[0])
    padded_tiles = tiles + [blank_tile] * (-len(tiles) % column_count)
    return np.vstack([np.hstack(padded_tiles[row_start:row_start + column_count]) for row_start in range(0, len(padded_tiles), column_count)])


def save_rgb_sheet(sheet_rgb: np.ndarray, sheet_path: Path) -> None:
    """Write an RGB uint8 sheet as PNG."""
    cv2.imwrite(str(sheet_path), cv2.cvtColor(sheet_rgb, cv2.COLOR_RGB2BGR))
    logger.info("wrote %s", sheet_path)


def single_type_config(light_type: LightSourceType) -> GlareSamplingConfig:
    """Training distribution narrowed to one procedural source of one light type, never glare-free."""
    return dataclasses.replace(GlareSamplingConfig(), no_glare_probability=0.0, light_type_probabilities={light_type: 1.0},
                               source_count_probabilities={1: 1.0}, hdri_reflection_share=0.0)


def hdri_only_config(hdri_category: str) -> GlareSamplingConfig:
    """Training distribution narrowed to pure image-based reflections from one HDRI category, never glare-free."""
    return dataclasses.replace(GlareSamplingConfig(), no_glare_probability=0.0, hdri_reflection_share=1.0, hdri_extra_emitter_probability=0.0,
                               hdri_category_probabilities={hdri_category: 1.0})


def render_sample_grid(review_crops: list[ReviewSourceCrop], sampling_config: GlareSamplingConfig, sample_count: int, seed_key: list[int],
                       label_prefix: str) -> np.ndarray:
    """Grid of `sample_count` glared crops drawn from a config, cycling over the review crops."""
    tiles = []
    for sample_index in range(sample_count):
        review_crop = review_crops[sample_index % len(review_crops)]
        data_dict = render_lens_glare_sample(review_crop.clean_eye_crop, review_crop.lens_label_mask, np.random.default_rng(seed_key + [sample_index]),
                                             sampling_config, review_crop.source_id)
        tiles.append(labeled_tile(data_dict["glare_eye_crop"], f"{label_prefix} #{sample_index} mask={data_dict['glare_mask'].mean():.2f}"))
    return tile_grid(tiles, LIGHT_TYPE_GRID_COLUMNS)


def render_hdri_and_default_sheets(review_crops: list[ReviewSourceCrop], output_directory: Path, seed: int) -> None:
    """Per-category HDRI grids and the default-distribution grid."""
    for category_index, hdri_category in enumerate(HDRI_CATEGORIES):
        save_rgb_sheet(render_sample_grid(review_crops, hdri_only_config(hdri_category), SAMPLES_PER_LIGHT_TYPE, [seed, 4000, category_index],
                                          f"hdri {hdri_category}"), output_directory / f"hdri__{hdri_category}.png")
    save_rgb_sheet(render_sample_grid(review_crops, GlareSamplingConfig(), DEFAULT_DISTRIBUTION_SAMPLE_COUNT, [seed, 5000], "default"),
                   output_directory / "default_distribution.png")


def render_light_type_sheets(review_crops: list[ReviewSourceCrop], output_directory: Path, seed: int) -> None:
    """Sheet (a): a grid of random samples for each light type."""
    for type_index, light_type in enumerate(LightSourceType):
        type_config = single_type_config(light_type)
        tiles = []
        for sample_index in range(SAMPLES_PER_LIGHT_TYPE):
            review_crop = review_crops[sample_index % len(review_crops)]
            random_generator = np.random.default_rng([seed, type_index, sample_index])
            data_dict = render_lens_glare_sample(review_crop.clean_eye_crop, review_crop.lens_label_mask, random_generator, type_config, review_crop.source_id)
            tiles.append(labeled_tile(data_dict["glare_eye_crop"], f"{light_type.value} #{sample_index}"))
        save_rgb_sheet(tile_grid(tiles, LIGHT_TYPE_GRID_COLUMNS), output_directory / f"light_type__{light_type.value}.png")


def render_mask_sheet(review_crops: list[ReviewSourceCrop], output_directory: Path, seed: int) -> None:
    """Sheet (b): clean | glared | glare_mask | lost_detail_mask rows from the default distribution."""
    rows = []
    for row_index in range(MASK_SHEET_ROW_COUNT):
        review_crop = review_crops[row_index % len(review_crops)]
        random_generator = np.random.default_rng([seed, 1000, row_index])
        data_dict = render_lens_glare_sample(review_crop.clean_eye_crop, review_crop.lens_label_mask, random_generator, GlareSamplingConfig(), review_crop.source_id)
        rows.append(np.hstack([
            labeled_tile(data_dict["clean_eye_crop"], f"clean {review_crop.source_id}"),
            labeled_tile(data_dict["glare_eye_crop"], "glared input"),
            labeled_tile(data_dict["glare_mask"], f"glare_mask mean={data_dict['glare_mask'].mean():.3f}"),
            labeled_tile(data_dict["lost_detail_mask"], f"lost_detail_mask mean={data_dict['lost_detail_mask'].mean():.3f}"),
        ]))
    save_rgb_sheet(np.vstack(rows), output_directory / "clean_glared_masks.png")


def render_severity_ladder(review_crops: list[ReviewSourceCrop], output_directory: Path, seed: int) -> None:
    """Sheet (c): per crop and light type, one fixed scene rendered at rising brightness."""
    ladder_types = (LightSourceType.RING_LIGHT, LightSourceType.WINDOW, LightSourceType.SOFTBOX, LightSourceType.SKY_WASH)
    rows = []
    for row_index, light_type in enumerate(ladder_types):
        review_crop = review_crops[row_index % len(review_crops)]
        lens_geometries = analyze_lens_label_mask(review_crop.lens_label_mask)
        ladder_config = dataclasses.replace(single_type_config(light_type), severity_probabilities={GlareSeverity.STRONG: 1.0})
        ladder_config = dataclasses.replace(ladder_config, veil_probability=0.0)
        scene = sample_glare_scene(lens_geometries, np.random.default_rng([seed, 2000, row_index]), ladder_config, hdri_library_for_config(ladder_config))
        for light_source in scene.light_sources:
            light_source.peak_linear = 1.0
        ladder_tiles = []
        for intensity_scale in SEVERITY_LADDER_SCALES:
            data_dict = render_glare_scene(review_crop.clean_eye_crop, review_crop.lens_label_mask, lens_geometries, scene, review_crop.source_id, intensity_scale)
            ladder_tiles.append(labeled_tile(data_dict["glare_eye_crop"], f"{light_type.value} peak={intensity_scale:g}"))
        rows.append(np.hstack(ladder_tiles))
    save_rgb_sheet(np.vstack(rows), output_directory / "severity_ladder.png")


def render_real_versus_synthetic_sheet(review_crops: list[ReviewSourceCrop], real_reference_directory: Path, output_directory: Path, seed: int) -> None:
    """Sheet (d): each real glare crop, then synthetic samples of the matching light type on its clean twin."""
    crops_by_name = {review_crop.source_id: review_crop for review_crop in review_crops}
    rows = []
    for real_file_name, (hdri_category, clean_twin_name) in REAL_REFERENCE_LOOKALIKES.items():
        real_crop_path = real_reference_directory / real_file_name
        if not real_crop_path.exists() or clean_twin_name not in crops_by_name:
            continue
        clean_twin = crops_by_name[clean_twin_name]
        row_tiles = [labeled_tile(load_rgb_float_image(real_crop_path), f"REAL {clean_twin_name}")]
        for sample_index in range(LOOKALIKES_PER_REFERENCE):
            random_generator = np.random.default_rng([seed, 3000, sample_index, len(rows)])
            data_dict = render_lens_glare_sample(clean_twin.clean_eye_crop, clean_twin.lens_label_mask, random_generator, hdri_only_config(hdri_category), clean_twin.source_id)
            row_tiles.append(labeled_tile(data_dict["glare_eye_crop"], f"synthetic hdri {hdri_category} #{sample_index}"))
        rows.append(np.hstack(row_tiles))
    if rows:
        save_rgb_sheet(np.vstack(rows), output_directory / "real_vs_synthetic.png")


def load_review_crops(manifest_path: Path, development_pairs_directory: Path, seed: int) -> list[ReviewSourceCrop]:
    """Prefer real manifest sources; fall back to the development crop pairs."""
    if manifest_path.exists():
        return load_crops_from_source_manifest(manifest_path, MANIFEST_REVIEW_CROP_COUNT, np.random.default_rng(seed))
    logger.warning("source manifest %s not found; using development crops in %s", manifest_path, development_pairs_directory)
    return load_crop_pairs_from_directory(development_pairs_directory)


def main() -> None:
    """Parse arguments and write every review sheet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_DIRECTORY)
    argument_parser.add_argument("--manifest-path", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH)
    argument_parser.add_argument("--development-pairs-dir", type=Path, default=DEFAULT_DEVELOPMENT_PAIRS_DIRECTORY)
    argument_parser.add_argument("--real-reference-dir", type=Path, default=None, help="directory of real glare eye crops (private, never in the repo)")
    argument_parser.add_argument("--seed", type=int, default=0)
    arguments = argument_parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    review_crops = load_review_crops(arguments.manifest_path, arguments.development_pairs_dir, arguments.seed)
    render_light_type_sheets(review_crops, arguments.output_dir, arguments.seed)
    render_hdri_and_default_sheets(review_crops, arguments.output_dir, arguments.seed)
    render_mask_sheet(review_crops, arguments.output_dir, arguments.seed)
    render_severity_ladder(review_crops, arguments.output_dir, arguments.seed)
    if arguments.real_reference_dir is not None:
        render_real_versus_synthetic_sheet(review_crops, arguments.real_reference_dir, arguments.output_dir, arguments.seed)


if __name__ == "__main__":
    main()
