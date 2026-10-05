"""Tests for the existing-glare score on synthetic faces: skin, two lenses, dark eyes, and painted glare."""

import cv2
import numpy as np

from training_sources.labeling.source_glare_score import SOURCE_GLARE_SCORE_THRESHOLD, compute_source_glare_score

IMAGE_SIDE = 1024
LEFT_EYE_XY = np.array([390.0, 490.0])
RIGHT_EYE_XY = np.array([630.0, 490.0])
SKIN_RGB = (200, 150, 120)


def make_synthetic_face_with_lenses() -> tuple[np.ndarray, np.ndarray]:
    """Return (photo RGB uint8, lens label map): flat skin, slightly darker lenses, dark eyes, light noise."""
    random_generator = np.random.default_rng(0)
    photo_rgb = np.empty((IMAGE_SIDE, IMAGE_SIDE, 3), dtype=np.float32)
    photo_rgb[:] = SKIN_RGB
    lens_label_map = np.zeros((IMAGE_SIDE, IMAGE_SIDE), dtype=np.uint8)
    for lens_value, eye_xy in ((1, LEFT_EYE_XY), (2, RIGHT_EYE_XY)):
        cv2.ellipse(lens_label_map, (int(eye_xy[0]), int(eye_xy[1]) + 5), (95, 60), 0, 0, 360, lens_value, thickness=cv2.FILLED)
    photo_rgb[lens_label_map > 0] *= 0.95  # lens transmission
    for eye_xy in (LEFT_EYE_XY, RIGHT_EYE_XY):
        cv2.ellipse(photo_rgb, (int(eye_xy[0]), int(eye_xy[1])), (40, 18), 0, 0, 360, (60, 45, 40), thickness=cv2.FILLED)
        cv2.circle(photo_rgb, (int(eye_xy[0]) + 5, int(eye_xy[1]) - 4), 4, (255, 255, 255), thickness=cv2.FILLED)  # catchlight
    photo_rgb += random_generator.normal(0, 3, photo_rgb.shape)
    return np.clip(photo_rgb, 0, 255).astype(np.uint8), lens_label_map


def test_clean_lenses_score_below_threshold() -> None:
    """No added light: the score stays under the threshold (catchlights in the eyes do not count)."""
    photo_rgb, lens_label_map = make_synthetic_face_with_lenses()
    glare_score, per_lens_features = compute_source_glare_score(photo_rgb, lens_label_map, LEFT_EYE_XY, RIGHT_EYE_XY)
    assert len(per_lens_features) == 2
    assert glare_score <= SOURCE_GLARE_SCORE_THRESHOLD


def test_bright_whitish_blob_raises_score_above_threshold() -> None:
    """A large white reflection on one lens (outside the eye opening) is detected as glare."""
    photo_rgb, lens_label_map = make_synthetic_face_with_lenses()
    cv2.rectangle(photo_rgb, (340, 515), (440, 545), (250, 250, 250), thickness=cv2.FILLED)
    photo_rgb[lens_label_map == 0] = make_synthetic_face_with_lenses()[0][lens_label_map == 0]
    glare_score, per_lens_features = compute_source_glare_score(photo_rgb, lens_label_map, LEFT_EYE_XY, RIGHT_EYE_XY)
    assert glare_score > SOURCE_GLARE_SCORE_THRESHOLD
    assert max(features.blob_fraction for features in per_lens_features) > 0.1


def test_brighter_skin_of_the_same_colour_is_not_glare() -> None:
    """Lit skin under the lens keeps the skin's chroma, so it must not count as a blob."""
    photo_rgb, lens_label_map = make_synthetic_face_with_lenses()
    lit_patch = photo_rgb[515:545, 340:440].astype(np.float32) * 1.25
    photo_rgb[515:545, 340:440] = np.clip(lit_patch, 0, 255).astype(np.uint8)
    _, per_lens_features = compute_source_glare_score(photo_rgb, lens_label_map, LEFT_EYE_XY, RIGHT_EYE_XY)
    assert max(features.blob_fraction for features in per_lens_features) < 0.01


def test_veil_over_whole_lens_raises_score() -> None:
    """A uniform additive wash over both lenses (lens lighter than the surrounding skin) is a veil."""
    photo_rgb, lens_label_map = make_synthetic_face_with_lenses()
    veiled_photo = photo_rgb.astype(np.float32)
    veiled_photo[lens_label_map > 0] += 45.0
    glare_score, per_lens_features = compute_source_glare_score(
        np.clip(veiled_photo, 0, 255).astype(np.uint8), lens_label_map, LEFT_EYE_XY, RIGHT_EYE_XY
    )
    assert min(features.veil_lift for features in per_lens_features) > 0.1
    assert glare_score > SOURCE_GLARE_SCORE_THRESHOLD


def test_no_lenses_scores_zero() -> None:
    """An empty lens map (no-glasses face) yields score 0 and no features."""
    photo_rgb, _ = make_synthetic_face_with_lenses()
    glare_score, per_lens_features = compute_source_glare_score(photo_rgb, np.zeros((IMAGE_SIDE, IMAGE_SIDE), np.uint8), LEFT_EYE_XY, RIGHT_EYE_XY)
    assert glare_score == 0.0
    assert per_lens_features == []
