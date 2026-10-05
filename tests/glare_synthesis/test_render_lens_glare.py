"""Invariants of the lens-glare renderer: determinism, data_dict contract, identity, locality, masks."""

import dataclasses

import cv2
import numpy as np
import pytest

from glare_synthesis.glare_masks import CLIPPED_SRGB_LEVEL, GLARE_MASK_CHANGE_HIGH, GLARE_MASK_CHANGE_LOW
from glare_synthesis.glare_sampling_config import GlareSamplingConfig, GlareSeverity, LightSourceType
from glare_synthesis.render_lens_glare import render_lens_glare_sample

SEEDS_PER_CHECK = 25
DATA_DICT_ARRAY_KEYS = ("clean_eye_crop", "glare_eye_crop", "glare_mask", "lost_detail_mask", "lens_mask")
# Settings that make the input a pure function of the glare: no grain, no JPEG, no bloom.
NOISE_FREE_OVERRIDES = {"glare_noise_scale_range": (0.0, 0.0), "jpeg_probability": 0.0, "bloom_probability": 0.0}
ALWAYS_GLARE_CONFIG = GlareSamplingConfig(no_glare_probability=0.0)
NOISE_FREE_CONFIG = dataclasses.replace(ALWAYS_GLARE_CONFIG, **NOISE_FREE_OVERRIDES)


def build_synthetic_eye_crop(crop_width: int, crop_height: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A textured skin-toned crop with dark irises and two elliptical lenses (label mask 1 and 2)."""
    random_generator = np.random.default_rng(seed)
    coarse_texture = random_generator.random((8, 16, 3)).astype(np.float32)
    skin = np.array([0.78, 0.6, 0.5], np.float32) * (0.8 + 0.25 * cv2.resize(coarse_texture, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC))
    skin += random_generator.normal(0.0, 0.01, skin.shape).astype(np.float32)
    lens_label_mask = np.zeros((crop_height, crop_width), np.uint8)
    for lens_label, center_fraction_x in ((1, 0.32), (2, 0.68)):
        center = (int(center_fraction_x * crop_width), int(0.5 * crop_height))
        cv2.ellipse(skin, center, (int(0.025 * crop_width), int(0.025 * crop_width)), 0, 0, 360, (0.12, 0.08, 0.06), -1)
        cv2.ellipse(lens_label_mask, center, (int(0.14 * crop_width), int(0.2 * crop_height)), 0, 0, 360, lens_label, -1)
    return np.clip(skin, 0.0, 1.0).astype(np.float32), lens_label_mask


@pytest.fixture(scope="module")
def default_crop() -> tuple[np.ndarray, np.ndarray]:
    """The 512x256 synthetic crop used by most tests."""
    return build_synthetic_eye_crop(512, 256)


def test_same_seed_gives_identical_sample(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    for seed in range(5):
        first = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), ALWAYS_GLARE_CONFIG)
        second = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), ALWAYS_GLARE_CONFIG)
        for key in DATA_DICT_ARRAY_KEYS:
            np.testing.assert_array_equal(first[key], second[key])
    different = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(99), ALWAYS_GLARE_CONFIG)
    assert not np.array_equal(first["glare_eye_crop"], different["glare_eye_crop"])


@pytest.mark.parametrize("crop_width, crop_height", [(512, 256), (1024, 512), (300, 150)])
def test_data_dict_shapes_dtypes_and_ranges(crop_width, crop_height):
    clean_eye_crop, lens_label_mask = build_synthetic_eye_crop(crop_width, crop_height)
    for seed in range(8):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), ALWAYS_GLARE_CONFIG, "face_7")
        assert data_dict["source_id"] == "face_7"
        for key in DATA_DICT_ARRAY_KEYS:
            expected_channels = 3 if key.endswith("crop") else 1
            assert data_dict[key].shape == (crop_height, crop_width, expected_channels), key
            assert data_dict[key].dtype == np.float32, key
            assert data_dict[key].min() >= 0.0 and data_dict[key].max() <= 1.0, key
        np.testing.assert_array_equal(data_dict["clean_eye_crop"], clean_eye_crop)
        np.testing.assert_array_equal(data_dict["lens_mask"][..., 0], (lens_label_mask > 0).astype(np.float32))


def test_glare_free_samples_are_identical_with_empty_masks(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    never_glare_config = GlareSamplingConfig(no_glare_probability=1.0)
    data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(0), never_glare_config)
    np.testing.assert_array_equal(data_dict["glare_eye_crop"], data_dict["clean_eye_crop"])
    assert not data_dict["glare_mask"].any() and not data_dict["lost_detail_mask"].any()
    no_lens_dict = render_lens_glare_sample(clean_eye_crop, np.zeros_like(lens_label_mask), np.random.default_rng(0), ALWAYS_GLARE_CONFIG)
    np.testing.assert_array_equal(no_lens_dict["glare_eye_crop"], clean_eye_crop)
    assert not no_lens_dict["glare_mask"].any() and not no_lens_dict["lens_mask"].any()


def test_default_glare_free_share_matches_config(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    sample_count = 400
    glare_free_count = sum(
        np.array_equal(render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed))["glare_eye_crop"], clean_eye_crop)
        for seed in range(sample_count)
    )
    # Default 12%; binomial sd at n=400 is ~1.6%, and a few faint samples may also round to identity.
    assert 0.07 < glare_free_count / sample_count < 0.2


def test_glare_never_leaves_lens_without_bloom_or_jpeg(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    outside_lens = lens_label_mask == 0
    for seed in range(SEEDS_PER_CHECK):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), NOISE_FREE_CONFIG)
        np.testing.assert_array_equal(data_dict["glare_eye_crop"][outside_lens], clean_eye_crop[outside_lens])


def test_changes_outside_lens_stay_within_bloom_and_jpeg_reach(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    lens_width = 0.28 * 512
    maximum_bloom_sigma = GlareSamplingConfig().bloom_sigma_fraction_of_lens_width_range[1] * lens_width
    reach_pixels = int(np.ceil(3.0 * maximum_bloom_sigma)) + 16 + 4  # bloom tail + one JPEG block + photo blur
    distance_to_lens = cv2.distanceTransform((lens_label_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    far_from_lens = distance_to_lens > reach_pixels
    for seed in range(SEEDS_PER_CHECK):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), ALWAYS_GLARE_CONFIG)
        np.testing.assert_array_equal(data_dict["glare_eye_crop"][far_from_lens], clean_eye_crop[far_from_lens])


def test_glare_mask_follows_the_pixel_difference_exactly_when_noise_free(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    for seed in range(SEEDS_PER_CHECK):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), NOISE_FREE_CONFIG)
        largest_change = np.abs(data_dict["glare_eye_crop"] - clean_eye_crop).max(axis=-1)
        glare_mask = data_dict["glare_mask"][..., 0]
        assert np.all(glare_mask[largest_change <= GLARE_MASK_CHANGE_LOW] == 0.0)
        assert np.all(glare_mask[largest_change >= GLARE_MASK_CHANGE_HIGH + 1e-4] == 1.0)


def test_glare_mask_agrees_with_pixel_difference_under_noise_and_jpeg(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    for seed in range(SEEDS_PER_CHECK):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), ALWAYS_GLARE_CONFIG)
        largest_change = np.abs(data_dict["glare_eye_crop"] - clean_eye_crop).max(axis=-1)
        glare_mask = data_dict["glare_mask"][..., 0]
        if (glare_mask == 1.0).sum() > 50:
            assert np.median(largest_change[glare_mask == 1.0]) > GLARE_MASK_CHANGE_HIGH * 0.8
        assert np.percentile(largest_change[glare_mask == 0.0], 99) < 0.04


def test_lost_detail_is_subset_of_glare_and_covers_clipped_pixels(default_crop):
    clean_eye_crop, lens_label_mask = default_crop
    strong_config = dataclasses.replace(NOISE_FREE_CONFIG, severity_probabilities={GlareSeverity.STRONG: 1.0},
                                        light_type_probabilities={LightSourceType.SOFTBOX: 1.0})
    clipped_pixels_seen = 0
    for seed in range(SEEDS_PER_CHECK):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), strong_config)
        glare_mask, lost_detail_mask = data_dict["glare_mask"][..., 0], data_dict["lost_detail_mask"][..., 0]
        assert np.all(lost_detail_mask <= glare_mask)
        newly_clipped = (data_dict["glare_eye_crop"] >= CLIPPED_SRGB_LEVEL).all(axis=-1) & (clean_eye_crop < 0.99).any(axis=-1)
        clipped_pixels_seen += int(newly_clipped.sum())
        np.testing.assert_allclose(lost_detail_mask[newly_clipped], glare_mask[newly_clipped])
    assert clipped_pixels_seen > 0
