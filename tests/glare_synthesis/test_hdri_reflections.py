"""Image-based (HDRI) reflections: aiming math, determinism, confinement to the lens, procedural-only mode."""

import dataclasses

import numpy as np
import pytest

from glare_synthesis.glare_sampling_config import GlareSamplingConfig
from glare_synthesis.hdri_reflection_rendering import (
    camera_to_environment_directions,
    environment_directions_to_lon_lat,
    solve_environment_rotation_for_target,
)
from glare_synthesis.render_lens_glare import render_lens_glare_sample
from test_render_lens_glare import NOISE_FREE_OVERRIDES, build_synthetic_eye_crop

HDRI_LIBRARY_AVAILABLE = GlareSamplingConfig().hdri_baked_manifest_path.exists()
requires_hdri_library = pytest.mark.skipif(not HDRI_LIBRARY_AVAILABLE, reason="baked HDRI library not on this machine")
HDRI_ONLY_CONFIG = GlareSamplingConfig(no_glare_probability=0.0, hdri_reflection_share=1.0, hdri_extra_emitter_probability=0.0)
AIM_TOLERANCE_RADIANS = 1e-3


def test_environment_rotation_puts_emitter_on_target_direction():
    random_generator = np.random.default_rng(0)
    checked_count = 0
    for _ in range(500):
        target_direction = random_generator.normal(size=3)
        target_direction[2] = -abs(target_direction[2]) - 0.5
        target_direction /= np.linalg.norm(target_direction)
        emitter_lon_lat = np.array([random_generator.uniform(-np.pi, np.pi), random_generator.uniform(-1.0, 1.0)])
        if abs(np.sin(emitter_lon_lat[1])) > np.hypot(target_direction[1], target_direction[2]):
            continue  # latitude unreachable by pitch alone; aiming is approximate there by design
        yaw, pitch = solve_environment_rotation_for_target(target_direction, emitter_lon_lat)
        if abs(pitch) >= np.radians(69.9):
            continue
        longitude, latitude = environment_directions_to_lon_lat(*camera_to_environment_directions(*target_direction.astype(np.float32), yaw, pitch))
        assert abs((longitude - emitter_lon_lat[0] + np.pi) % (2 * np.pi) - np.pi) < AIM_TOLERANCE_RADIANS
        assert abs(latitude - emitter_lon_lat[1]) < AIM_TOLERANCE_RADIANS
        checked_count += 1
    assert checked_count > 300


@requires_hdri_library
def test_hdri_samples_are_deterministic_and_visible():
    clean_eye_crop, lens_label_mask = build_synthetic_eye_crop(512, 256)
    visible_count = 0
    for seed in range(12):
        first = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), HDRI_ONLY_CONFIG)
        second = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), HDRI_ONLY_CONFIG)
        np.testing.assert_array_equal(first["glare_eye_crop"], second["glare_eye_crop"])
        visible_count += int(first["glare_mask"].mean() > 0.01)
    assert visible_count >= 6


@requires_hdri_library
def test_hdri_reflection_stays_inside_the_lens():
    clean_eye_crop, lens_label_mask = build_synthetic_eye_crop(512, 256)
    noise_free_hdri_config = dataclasses.replace(HDRI_ONLY_CONFIG, **NOISE_FREE_OVERRIDES)
    outside_lens = lens_label_mask == 0
    for seed in range(12):
        data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(seed), noise_free_hdri_config)
        np.testing.assert_array_equal(data_dict["glare_eye_crop"][outside_lens], clean_eye_crop[outside_lens])


def test_procedural_only_mode_needs_no_hdri_library(tmp_path):
    clean_eye_crop, lens_label_mask = build_synthetic_eye_crop(512, 256)
    procedural_config = GlareSamplingConfig(no_glare_probability=0.0, hdri_reflection_share=0.0, hdri_baked_manifest_path=tmp_path / "missing.jsonl")
    data_dict = render_lens_glare_sample(clean_eye_crop, lens_label_mask, np.random.default_rng(3), procedural_config)
    assert data_dict["glare_eye_crop"].shape == clean_eye_crop.shape
