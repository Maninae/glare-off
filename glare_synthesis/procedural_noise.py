"""Cheap smooth random fields: upsampled random grids, used for textures, wear and scenes."""

import cv2
import numpy as np


def smooth_random_field(
    random_generator: np.random.Generator,
    output_height: int,
    output_width: int,
    grid_cells: int,
    channels: int = 1,
) -> np.ndarray:
    """Return a smooth float32 field in roughly [0, 1] by bicubic-upsampling a coarse random grid.

    `grid_cells` sets the feature size: about output_size / grid_cells pixels per blob.
    Shape is (H, W) for one channel, else (H, W, channels).
    """
    grid_shape = (max(2, grid_cells), max(2, grid_cells)) if channels == 1 else (max(2, grid_cells), max(2, grid_cells), channels)
    coarse_grid = random_generator.random(grid_shape).astype(np.float32)
    smooth_field = cv2.resize(coarse_grid, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
    return np.clip(smooth_field, 0.0, 1.0)


def fractal_random_field(
    random_generator: np.random.Generator,
    output_height: int,
    output_width: int,
    base_grid_cells: int,
    octave_count: int = 3,
) -> np.ndarray:
    """Sum of smooth fields at doubling frequencies with halving amplitude, rescaled to [0, 1]."""
    accumulated_field = np.zeros((output_height, output_width), np.float32)
    amplitude = 1.0
    for octave_index in range(octave_count):
        accumulated_field += amplitude * smooth_random_field(random_generator, output_height, output_width, base_grid_cells * 2**octave_index)
        amplitude *= 0.5
    field_min, field_max = float(accumulated_field.min()), float(accumulated_field.max())
    return (accumulated_field - field_min) / max(field_max - field_min, 1e-6)
