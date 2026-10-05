"""Measure single-core render time per sample (the DataLoader-worker budget is 40 ms at 512x256).

    python -m glare_synthesis.benchmark_render_speed --sample-count 300
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from glare_synthesis.render_lens_glare import render_lens_glare_sample
from glare_synthesis.render_review_sheets import DEFAULT_DEVELOPMENT_PAIRS_DIRECTORY, load_review_crops
from glare_synthesis.review_source_crops import DEFAULT_SOURCE_MANIFEST_PATH

logger = logging.getLogger(__name__)

WARMUP_SAMPLE_COUNT = 10
UPSCALED_RESOLUTION_FACTOR = 2


def time_render_calls(clean_crops: list[np.ndarray], lens_masks: list[np.ndarray], sample_count: int) -> np.ndarray:
    """Render `sample_count` samples cycling over the crops; return per-sample milliseconds."""
    for warmup_index in range(WARMUP_SAMPLE_COUNT):
        render_lens_glare_sample(clean_crops[0], lens_masks[0], np.random.default_rng(warmup_index))
    elapsed_milliseconds = []
    for sample_index in range(sample_count):
        crop_index = sample_index % len(clean_crops)
        start_time = time.perf_counter()
        render_lens_glare_sample(clean_crops[crop_index], lens_masks[crop_index], np.random.default_rng(sample_index))
        elapsed_milliseconds.append((time.perf_counter() - start_time) * 1000.0)
    return np.array(elapsed_milliseconds)


def log_timing_summary(resolution_label: str, elapsed_milliseconds: np.ndarray) -> None:
    """Log mean / median / p90 / max milliseconds per sample."""
    logger.info("%s: mean %.1f ms, median %.1f, p90 %.1f, max %.1f (n=%d, 1 thread)", resolution_label, elapsed_milliseconds.mean(),
                np.median(elapsed_milliseconds), np.percentile(elapsed_milliseconds, 90), elapsed_milliseconds.max(), len(elapsed_milliseconds))


def main() -> None:
    """Benchmark at the default crop size and at 2x."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--sample-count", type=int, default=300)
    argument_parser.add_argument("--manifest-path", type=Path, default=DEFAULT_SOURCE_MANIFEST_PATH)
    argument_parser.add_argument("--development-pairs-dir", type=Path, default=DEFAULT_DEVELOPMENT_PAIRS_DIRECTORY)
    arguments = argument_parser.parse_args()
    cv2.setNumThreads(1)
    review_crops = load_review_crops(arguments.manifest_path, arguments.development_pairs_dir, seed=0)
    clean_crops = [review_crop.clean_eye_crop for review_crop in review_crops]
    lens_masks = [review_crop.lens_label_mask for review_crop in review_crops]
    crop_height, crop_width = clean_crops[0].shape[:2]
    log_timing_summary(f"{crop_width}x{crop_height}", time_render_calls(clean_crops, lens_masks, arguments.sample_count))
    upscaled_size = (crop_width * UPSCALED_RESOLUTION_FACTOR, crop_height * UPSCALED_RESOLUTION_FACTOR)
    upscaled_crops = [cv2.resize(clean_crop, upscaled_size, interpolation=cv2.INTER_CUBIC).clip(0.0, 1.0) for clean_crop in clean_crops]
    upscaled_masks = [cv2.resize(lens_mask, upscaled_size, interpolation=cv2.INTER_NEAREST) for lens_mask in lens_masks]
    log_timing_summary(f"{upscaled_size[0]}x{upscaled_size[1]}", time_render_calls(upscaled_crops, upscaled_masks, arguments.sample_count // 3))


if __name__ == "__main__":
    main()
