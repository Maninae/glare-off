"""Tile eye crops into a captioned contact sheet PNG (for eyeballing the cache and the misclassified faces)."""

from pathlib import Path

import cv2
import numpy as np

TILE_WIDTH = 384
TILE_HEIGHT = 192
CAPTION_HEIGHT = 22
COLUMN_COUNT = 4
CAPTION_FONT_SCALE = 0.5
BACKGROUND_LEVEL = 255


def write_eye_crop_contact_sheet(eye_crops_uint8: list[np.ndarray], captions: list[str], output_path: Path) -> Path:
    """Write the crops (each (H, W, 3) uint8 RGB) in a COLUMN_COUNT-wide grid with one caption line under each.

    Returns the output path. An empty list writes a small sheet that says so.
    """
    if not eye_crops_uint8:
        eye_crops_uint8, captions = [np.full((TILE_HEIGHT, TILE_WIDTH, 3), BACKGROUND_LEVEL, np.uint8)], ["(nothing to show)"]
    row_count = (len(eye_crops_uint8) + COLUMN_COUNT - 1) // COLUMN_COUNT
    cell_height = TILE_HEIGHT + CAPTION_HEIGHT
    sheet = np.full((row_count * cell_height, COLUMN_COUNT * TILE_WIDTH, 3), BACKGROUND_LEVEL, dtype=np.uint8)
    for tile_index, (eye_crop, caption) in enumerate(zip(eye_crops_uint8, captions)):
        top = (tile_index // COLUMN_COUNT) * cell_height
        left = (tile_index % COLUMN_COUNT) * TILE_WIDTH
        sheet[top : top + TILE_HEIGHT, left : left + TILE_WIDTH] = cv2.resize(eye_crop, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, caption, (left + 4, top + TILE_HEIGHT + 16), cv2.FONT_HERSHEY_SIMPLEX, CAPTION_FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return output_path
