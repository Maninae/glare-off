"""Tests for deterministic split hashing and left/right lens assignment on synthetic masks."""

from collections import Counter

import cv2
import numpy as np

from training_sources.labeling.lens_mask_assignment import LensLabel, LensMaskStatus, assign_left_right_lenses
from training_sources.source_split_assignment import (
    SourceSplit,
    assign_split_by_grouping_key,
    flickr_account_from_photo_url,
)

IMAGE_SIDE = 1024
LEFT_EYE_XY = np.array([390.0, 490.0])
RIGHT_EYE_XY = np.array([630.0, 490.0])
LENS_HALF_AXES = (95, 60)  # about 0.8d x 0.5d lenses for d = 240


def test_flickr_account_is_the_user_segment() -> None:
    """Photos from one account share a grouping key, case-insensitively."""
    assert flickr_account_from_photo_url("https://www.flickr.com/photos/SomeUser/123/") == "someuser"
    assert flickr_account_from_photo_url("https://www.flickr.com/photos/someuser/999/") == "someuser"
    assert flickr_account_from_photo_url("https://example.com/odd") == "https://example.com/odd"


def test_split_is_deterministic_and_roughly_90_5_5() -> None:
    """Same key always gets the same split; over many keys the proportions are close to 90/5/5."""
    assert assign_split_by_grouping_key("account-a") == assign_split_by_grouping_key("account-a")
    split_counts = Counter(assign_split_by_grouping_key(f"account-{key_index}") for key_index in range(20000))
    assert set(split_counts) == {SourceSplit.TRAIN, SourceSplit.VAL, SourceSplit.TEST}
    assert abs(split_counts[SourceSplit.TRAIN] / 20000 - 0.90) < 0.01
    assert abs(split_counts[SourceSplit.VAL] / 20000 - 0.05) < 0.01
    assert abs(split_counts[SourceSplit.TEST] / 20000 - 0.05) < 0.01


def draw_lens_ellipses(lens_centers: list[tuple[int, int]], half_axes: tuple[int, int] = LENS_HALF_AXES) -> np.ndarray:
    """Return a bool mask with one filled ellipse per lens center."""
    lens_mask = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    for lens_center in lens_centers:
        cv2.ellipse(lens_mask, lens_center, half_axes, 0, 0, 360, 1, thickness=cv2.FILLED)
    return lens_mask.astype(bool)


def test_two_separate_lenses_get_left_and_right_labels() -> None:
    """Each lens is labelled by which side of the face it is on in the image."""
    lens_mask = draw_lens_ellipses([(390, 495), (630, 495)])
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.status == LensMaskStatus.TWO_LENSES
    assert result.lens_label_map[495, 390] == LensLabel.IMAGE_LEFT_LENS
    assert result.lens_label_map[495, 630] == LensLabel.IMAGE_RIGHT_LENS
    assert result.lens_label_map[100, 100] == LensLabel.BACKGROUND


def test_lenses_merged_across_the_bridge_are_split_at_the_midline() -> None:
    """A segmentation joined over the nose bridge still yields two lenses, divided at the eye bisector."""
    lens_mask = draw_lens_ellipses([(390, 495), (630, 495)])
    lens_mask[480:500, 470:550] = True  # bridge joining the two lenses
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.status == LensMaskStatus.TWO_LENSES
    assert result.lens_label_map[490, 505] == LensLabel.IMAGE_LEFT_LENS
    assert result.lens_label_map[490, 515] == LensLabel.IMAGE_RIGHT_LENS


def test_holes_from_catchlights_are_filled() -> None:
    """A hole punched in a lens (e.g. a catchlight) becomes lens in the label map."""
    lens_mask = draw_lens_ellipses([(390, 495), (630, 495)])
    lens_mask[485:495, 385:395] = False
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.lens_label_map[490, 390] == LensLabel.IMAGE_LEFT_LENS


def test_stray_specks_are_ignored() -> None:
    """Small components far from the eyes do not become lenses."""
    lens_mask = draw_lens_ellipses([(390, 495), (630, 495)])
    lens_mask[900:905, 100:105] = True
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.lens_label_map[902, 102] == LensLabel.BACKGROUND


def test_one_lens_fails_for_frontal_but_passes_for_turned_head() -> None:
    """A single lens is only acceptable when the head is strongly turned."""
    lens_mask = draw_lens_ellipses([(390, 495)])
    frontal = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    turned = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=True)
    assert frontal.status == LensMaskStatus.FAILED
    assert not frontal.lens_label_map.any()
    assert turned.status == LensMaskStatus.SINGLE_LENS
    assert turned.lens_label_map[495, 390] == LensLabel.IMAGE_LEFT_LENS


def test_lens_far_from_its_eye_is_rejected() -> None:
    """A blob that does not cover or touch the eye is not that eye's lens."""
    lens_mask = draw_lens_ellipses([(390, 800), (630, 495)])
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.status == LensMaskStatus.FAILED
    assert "eye_outside_lens" in result.failure_reason


def test_non_convex_leak_is_rejected() -> None:
    """A lens with a long leak onto the skin fails the solidity check."""
    lens_mask = draw_lens_ellipses([(390, 495), (630, 495)])
    lens_mask[540:760, 300:330] = True  # leak running down the cheek
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.status == LensMaskStatus.FAILED
    assert "lens_not_convex" in result.failure_reason


def test_very_unequal_lenses_fail_for_frontal_face() -> None:
    """Two plausible but very different-sized lenses are implausible on a frontal face."""
    lens_mask = draw_lens_ellipses([(390, 495)], half_axes=(110, 75)) | draw_lens_ellipses([(630, 495)], half_axes=(55, 35))
    result = assign_left_right_lenses(lens_mask, LEFT_EYE_XY, RIGHT_EYE_XY, head_is_strongly_turned=False)
    assert result.status == LensMaskStatus.FAILED
    assert result.failure_reason == "asymmetric_lens_areas"
