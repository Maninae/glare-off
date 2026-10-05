"""The shared crop resampler: pre-shrink rule, geometry preserved through the pre-shrink, no aliasing."""

import cv2
import numpy as np
import pytest

from eye_crop.eye_crop_geometry import (
    compute_area_prefilter_factor,
    compute_photo_to_eye_crop_affine,
    extract_eye_crop_with_area_prefilter,
    eye_crop_scale,
    warp_eye_crop_layer_back_to_photo,
)

CROP_WIDTH = 128
CROP_HEIGHT = 64


def affine_for_scale(crop_scale: float, roll_degrees: float = 7.0, center_xy=(400.0, 300.0)) -> np.ndarray:
    """Affine whose crop scale is `crop_scale`, eyes rolled by `roll_degrees` about `center_xy`."""
    eye_distance = CROP_WIDTH * 0.36 / crop_scale
    half_offset = 0.5 * eye_distance * np.array([np.cos(np.radians(roll_degrees)), np.sin(np.radians(roll_degrees))])
    center = np.asarray(center_xy)
    return compute_photo_to_eye_crop_affine(center - half_offset, center + half_offset, CROP_WIDTH, CROP_HEIGHT)


@pytest.mark.parametrize("crop_scale, expected_factor", [(1.2, 1), (0.5, 1), (0.49, 2), (0.34, 2), (0.25, 4), (0.15, 6)])
def test_prefilter_factor_rule(crop_scale, expected_factor):
    assert compute_area_prefilter_factor(crop_scale) == expected_factor


def test_remaining_warp_scale_stays_in_half_to_one():
    for crop_scale in np.linspace(0.05, 0.4999, 400):
        remaining_scale = crop_scale * compute_area_prefilter_factor(crop_scale)
        assert 0.5 <= remaining_scale <= 1.0 + 1e-9, crop_scale


def test_no_prefilter_scale_is_a_plain_bilinear_warp():
    photo = np.random.default_rng(0).random((600, 800, 3)).astype(np.float32)
    affine = affine_for_scale(1.3)
    expected = cv2.warpAffine(photo, affine, (CROP_WIDTH, CROP_HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    np.testing.assert_array_equal(extract_eye_crop_with_area_prefilter(photo, affine, CROP_WIDTH, CROP_HEIGHT), expected)


@pytest.mark.parametrize("crop_scale", [0.45, 0.3, 0.17])
def test_prefiltered_crop_keeps_the_affine_geometry(crop_scale):
    """A smooth blob at a known photo point lands where the ORIGINAL affine maps it (to 0.15 crop px)."""
    affine = affine_for_scale(crop_scale)
    blob_photo_xy = np.array([413.0, 291.0])
    rows, columns = np.mgrid[0:800, 0:1000].astype(np.float32)
    blob_sigma = 4.0 / crop_scale  # 4 crop pixels wide whatever the scale
    photo = np.exp(-((columns - blob_photo_xy[0]) ** 2 + (rows - blob_photo_xy[1]) ** 2) / (2 * blob_sigma**2))
    crop = extract_eye_crop_with_area_prefilter(photo, affine, CROP_WIDTH, CROP_HEIGHT)
    crop_rows, crop_columns = np.mgrid[0:CROP_HEIGHT, 0:CROP_WIDTH]
    centroid = np.array([(crop * crop_columns).sum(), (crop * crop_rows).sum()]) / crop.sum()
    expected = affine @ np.append(blob_photo_xy, 1.0)
    assert np.abs(centroid - expected).max() < 0.15


def test_prefilter_removes_aliasing_of_fine_texture():
    """A 1-pixel checkerboard must average to flat gray, not alias into a random pattern."""
    rows, columns = np.mgrid[0:1200, 0:1600]
    checkerboard = ((rows + columns) % 2).astype(np.float32)
    affine = affine_for_scale(0.2, center_xy=(800.0, 600.0))
    prefiltered = extract_eye_crop_with_area_prefilter(checkerboard, affine, CROP_WIDTH, CROP_HEIGHT)
    plain_bilinear = cv2.warpAffine(checkerboard, affine, (CROP_WIDTH, CROP_HEIGHT), flags=cv2.INTER_LINEAR)
    assert prefiltered.std() < 0.05 and abs(prefiltered.mean() - 0.5) < 0.02
    assert plain_bilinear.std() > 0.1


def test_round_trip_through_the_original_affine_still_holds():
    """Crop with the prefilter, warp back with the unchanged affine: a smooth photo comes back."""
    rows, columns = np.mgrid[0:800, 0:1000].astype(np.float32)
    photo = 0.5 + 0.4 * np.sin(columns / 90.0) * np.cos(rows / 70.0)
    affine = affine_for_scale(0.3, center_xy=(500.0, 400.0))
    assert eye_crop_scale(affine) == pytest.approx(0.3)
    crop = extract_eye_crop_with_area_prefilter(photo, affine, CROP_WIDTH, CROP_HEIGHT)
    coverage = warp_eye_crop_layer_back_to_photo(np.ones_like(crop), affine, 1000, 800)
    warped_back = warp_eye_crop_layer_back_to_photo(crop, affine, 1000, 800)
    fully_covered = coverage > 0.999
    assert fully_covered.sum() > 10_000
    assert np.abs(warped_back - photo)[fully_covered].max() < 0.02
