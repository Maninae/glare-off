"""Render human-review contact sheets: aligned eye crops with the lens-mask outline drawn on.

Each tile is the contract's 512x256 eye crop (`eye_crop.eye_crop_geometry`) with:
- the image-left lens outlined in green and the image-right lens in cyan,
- for rejected faces, the raw (unaccepted) segmentation outlined in red when it was saved,
- a caption strip under the crop: source id, glare score, and a short status.

Sheets are grids of up to TILES_PER_SHEET tiles, SHEET_COLUMN_COUNT columns, saved as JPEG.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from eye_crop.eye_crop_geometry import EYE_CROP_HEIGHT, EYE_CROP_WIDTH, compute_photo_to_eye_crop_affine, extract_eye_crop

logger = logging.getLogger(__name__)

SHEET_COLUMN_COUNT = 4
TILES_PER_SHEET = 40
CAPTION_STRIP_HEIGHT = 34
TILE_GUTTER_PIXELS = 6
SHEET_JPEG_QUALITY = 88
OUTLINE_THICKNESS_PIXELS = 2
CAPTION_FONT_SCALE = 0.62
# BGR colours.
IMAGE_LEFT_LENS_OUTLINE_BGR = (60, 220, 60)
IMAGE_RIGHT_LENS_OUTLINE_BGR = (230, 210, 40)
RAW_SEGMENTATION_OUTLINE_BGR = (40, 40, 235)
CAPTION_BACKGROUND_BGR = (28, 28, 28)
CAPTION_TEXT_BGR = (235, 235, 235)
SHEET_BACKGROUND_BGR = (12, 12, 12)
# Median FFHQ eye centers (measured on our annotations), used to frame faces where YuNet found none.
FFHQ_TYPICAL_IMAGE_LEFT_EYE_XY = (389.0, 493.0)
FFHQ_TYPICAL_IMAGE_RIGHT_EYE_XY = (629.0, 492.0)


def draw_mask_outline(eye_crop_bgr: np.ndarray, crop_space_mask: np.ndarray, outline_bgr: tuple[int, int, int]) -> None:
    """Draw the outer contours of a crop-space bool mask onto the crop, in place."""
    contours, _ = cv2.findContours(crop_space_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(eye_crop_bgr, contours, -1, outline_bgr, OUTLINE_THICKNESS_PIXELS, lineType=cv2.LINE_AA)


def render_review_tile(
    photo_bgr: np.ndarray,
    image_left_eye_xy: np.ndarray | None,
    image_right_eye_xy: np.ndarray | None,
    lens_label_map: np.ndarray | None,
    raw_segmentation_mask: np.ndarray | None,
    caption_text: str,
) -> np.ndarray:
    """Return one (256 + caption, 512, 3) BGR tile: eye crop, lens outlines, caption.

    Missing eyes fall back to the typical FFHQ eye positions so the tile still shows the eye region.
    """
    if image_left_eye_xy is None or image_right_eye_xy is None:
        image_left_eye_xy, image_right_eye_xy = FFHQ_TYPICAL_IMAGE_LEFT_EYE_XY, FFHQ_TYPICAL_IMAGE_RIGHT_EYE_XY
    photo_to_crop_affine = compute_photo_to_eye_crop_affine(np.asarray(image_left_eye_xy), np.asarray(image_right_eye_xy))
    eye_crop_bgr = extract_eye_crop(photo_bgr, photo_to_crop_affine).copy()
    if raw_segmentation_mask is not None:
        raw_crop = extract_eye_crop(raw_segmentation_mask, photo_to_crop_affine, interpolation=cv2.INTER_NEAREST)
        draw_mask_outline(eye_crop_bgr, raw_crop > 0, RAW_SEGMENTATION_OUTLINE_BGR)
    if lens_label_map is not None:
        label_crop = extract_eye_crop(lens_label_map, photo_to_crop_affine, interpolation=cv2.INTER_NEAREST)
        draw_mask_outline(eye_crop_bgr, label_crop == 1, IMAGE_LEFT_LENS_OUTLINE_BGR)
        draw_mask_outline(eye_crop_bgr, label_crop == 2, IMAGE_RIGHT_LENS_OUTLINE_BGR)
    caption_strip = np.full((CAPTION_STRIP_HEIGHT, EYE_CROP_WIDTH, 3), CAPTION_BACKGROUND_BGR, dtype=np.uint8)
    cv2.putText(
        caption_strip, caption_text, (8, CAPTION_STRIP_HEIGHT - 11), cv2.FONT_HERSHEY_SIMPLEX,
        CAPTION_FONT_SCALE, CAPTION_TEXT_BGR, 1, lineType=cv2.LINE_AA,
    )
    return np.vstack([eye_crop_bgr, caption_strip])


def assemble_contact_sheet(tiles_bgr: list[np.ndarray], column_count: int = SHEET_COLUMN_COUNT) -> np.ndarray:
    """Lay equally sized tiles out in a grid with thin dark gutters."""
    tile_height = EYE_CROP_HEIGHT + CAPTION_STRIP_HEIGHT
    row_count = (len(tiles_bgr) + column_count - 1) // column_count
    sheet_height = row_count * tile_height + (row_count + 1) * TILE_GUTTER_PIXELS
    sheet_width = column_count * EYE_CROP_WIDTH + (column_count + 1) * TILE_GUTTER_PIXELS
    contact_sheet = np.full((sheet_height, sheet_width, 3), SHEET_BACKGROUND_BGR, dtype=np.uint8)
    for tile_index, tile_bgr in enumerate(tiles_bgr):
        row_index, column_index = divmod(tile_index, column_count)
        top = TILE_GUTTER_PIXELS + row_index * (tile_height + TILE_GUTTER_PIXELS)
        left = TILE_GUTTER_PIXELS + column_index * (EYE_CROP_WIDTH + TILE_GUTTER_PIXELS)
        contact_sheet[top : top + tile_height, left : left + EYE_CROP_WIDTH] = tile_bgr
    return contact_sheet


def load_review_tile_for_annotation(annotation_row: dict, caption_text: str, raw_segmentation_path: Path | None) -> np.ndarray | None:
    """Render the tile for one annotation row (None when the photo is not on disk)."""
    photo_bgr = cv2.imread(annotation_row.get("photo_path", ""), cv2.IMREAD_COLOR) if annotation_row.get("photo_path") else None
    if photo_bgr is None:
        return None
    lens_label_map = None
    if annotation_row.get("lens_mask_path"):
        lens_label_map = cv2.imread(annotation_row["lens_mask_path"], cv2.IMREAD_GRAYSCALE)
    raw_segmentation_mask = None
    if raw_segmentation_path is not None and raw_segmentation_path.exists():
        raw_segmentation_mask = cv2.imread(str(raw_segmentation_path), cv2.IMREAD_GRAYSCALE)
    return render_review_tile(
        photo_bgr,
        annotation_row.get("image_left_eye_xy"),
        annotation_row.get("image_right_eye_xy"),
        lens_label_map,
        raw_segmentation_mask,
        caption_text,
    )


def write_contact_sheets(tiles_bgr: list[np.ndarray], output_directory: Path, sheet_name_prefix: str) -> list[Path]:
    """Split tiles into sheets of TILES_PER_SHEET and save them as `<prefix>__sheet-NN.jpg`."""
    output_directory.mkdir(parents=True, exist_ok=True)
    written_paths = []
    for sheet_index, first_tile_index in enumerate(range(0, len(tiles_bgr), TILES_PER_SHEET), start=1):
        sheet_path = output_directory / f"{sheet_name_prefix}__sheet-{sheet_index:02d}.jpg"
        contact_sheet = assemble_contact_sheet(tiles_bgr[first_tile_index : first_tile_index + TILES_PER_SHEET])
        cv2.imwrite(str(sheet_path), contact_sheet, [cv2.IMWRITE_JPEG_QUALITY, SHEET_JPEG_QUALITY])
        written_paths.append(sheet_path)
    logger.info("wrote %d sheet(s) for %s", len(written_paths), sheet_name_prefix)
    return written_paths
