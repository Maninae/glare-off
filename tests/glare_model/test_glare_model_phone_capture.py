"""Simulated phone capture: contract shapes hold, and the lens mask stays on the lens through the re-sampling."""

import cv2
import numpy as np
import pytest
import torch

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, eye_crop_scale

from glare_model.data.crop_augmentation import EyeCenterJitterSettings, PhotometricJitterSettings
from glare_model.data.fake_data_sources import FakeSourceFaceProvider, synthesize_fake_blob_glare
from glare_model.data.glare_pair_dataset import GlarePairDataset, GlarePairDatasetSettings, lens_labels_to_one_hot_planes
from glare_model.data.phone_capture_simulation import PhoneCaptureSimulationSettings, apply_capture_degradation
from glare_model.data.source_face import LENS_LABEL_IMAGE_LEFT, LENS_LABEL_IMAGE_RIGHT, SourceFace

CROP_HEIGHT = 128
CROP_WIDTH = 256
ALWAYS_SIMULATE_CLEAN = PhoneCaptureSimulationSettings(probability=1.0, jpeg_probability=0.0, noise_probability=0.0)
ALWAYS_SIMULATE_DEGRADED = PhoneCaptureSimulationSettings(probability=1.0, jpeg_probability=1.0, noise_probability=1.0)


class KnownEllipseLensProvider:
    """One 1600x1200 photo: black, with two white lens ellipses that exactly match the lens labels."""

    def __len__(self) -> int:
        return 1

    def load_source_face(self, source_index: int) -> SourceFace:
        photo_rgb = np.zeros((1200, 1600, 3), np.float32)
        lens_label_mask = np.zeros((1200, 1600), np.uint8)
        left_eye, right_eye = np.array([700.0, 600.0]), np.array([900.0, 620.0])
        for eye_center, lens_label in ((left_eye, LENS_LABEL_IMAGE_LEFT), (right_eye, LENS_LABEL_IMAGE_RIGHT)):
            center = (int(eye_center[0]), int(eye_center[1]))
            cv2.ellipse(lens_label_mask, center, (80, 55), 6.0, 0, 360, int(lens_label), -1)
        photo_rgb[lens_label_mask > 0] = 1.0
        return SourceFace("ellipse_000000", photo_rgb, left_eye, right_eye, lens_label_mask)


def synthesize_no_glare(clean_eye_crop: np.ndarray, lens_label_mask: np.ndarray, random_generator: np.random.Generator) -> dict:
    """Identity synthesizer: glare == clean, empty masks, lens mask straight from the labels."""
    empty_mask = np.zeros(clean_eye_crop.shape[:2] + (1,), np.float32)
    return {
        "clean_eye_crop": clean_eye_crop,
        "glare_eye_crop": clean_eye_crop.copy(),
        "glare_mask": empty_mask,
        "lost_detail_mask": empty_mask.copy(),
        "lens_mask": (lens_label_mask > 0).astype(np.float32)[..., None],
        "source_id": "ellipse",
    }


def build_ellipse_dataset(phone_capture_settings: PhoneCaptureSimulationSettings, seed: int) -> GlarePairDataset:
    return GlarePairDataset(
        KnownEllipseLensProvider(),
        synthesize_no_glare,
        GlarePairDatasetSettings(crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH, fixed_seed=seed, use_photometric_augmentation=False),
        EyeCenterJitterSettings(),
        PhotometricJitterSettings(),
        phone_capture_settings,
    )


def test_phone_capture_samples_honor_the_synthesis_contract():
    dataset = GlarePairDataset(
        FakeSourceFaceProvider("train", face_count=6),
        synthesize_fake_blob_glare,
        GlarePairDatasetSettings(crop_height=CROP_HEIGHT, crop_width=CROP_WIDTH),
        EyeCenterJitterSettings(),
        PhotometricJitterSettings(),
        ALWAYS_SIMULATE_DEGRADED,
    )
    expected_channels = {"clean_eye_crop": 3, "glare_eye_crop": 3, "glare_mask": 1, "lost_detail_mask": 1, "lens_mask": 1}
    for item_index in range(len(dataset)):
        sample = dataset[item_index]
        for key, channel_count in expected_channels.items():
            assert sample[key].shape == (channel_count, CROP_HEIGHT, CROP_WIDTH), key
            assert sample[key].dtype == torch.float32, key
            assert 0.0 <= float(sample[key].min()) and float(sample[key].max()) <= 1.0, key


def test_lens_mask_stays_aligned_with_the_lens_pixels_at_phone_scale():
    """The white ellipses ARE the lenses: thresholded crop pixels must equal the lens mask."""
    for seed in range(12):
        sample = build_ellipse_dataset(ALWAYS_SIMULATE_CLEAN, seed)[0]
        lens_pixels_in_image = sample["clean_eye_crop"].mean(dim=0) > 0.5
        lens_mask = sample["lens_mask"][0] > 0.5
        assert lens_mask.sum() > 200, seed  # the lenses are in frame and not shrunk to nothing
        mismatch_fraction = float((lens_pixels_in_image != lens_mask).float().mean())
        assert mismatch_fraction < 0.002, (seed, mismatch_fraction)


def test_lens_mask_stays_aligned_with_jpeg_and_noise():
    for seed in range(6):
        sample = build_ellipse_dataset(ALWAYS_SIMULATE_DEGRADED, seed)[0]
        mismatch_fraction = float(((sample["clean_eye_crop"].mean(dim=0) > 0.5) != (sample["lens_mask"][0] > 0.5)).float().mean())
        assert mismatch_fraction < 0.01, (seed, mismatch_fraction)


def test_phone_capture_produces_the_drawn_crop_scale():
    """The re-sampled region + mapped eyes give a crop at the target scale, or the native one if already smaller."""
    dataset = build_ellipse_dataset(ALWAYS_SIMULATE_CLEAN, 0)
    source_face = KnownEllipseLensProvider().load_source_face(0)
    planes = lens_labels_to_one_hot_planes(source_face.lens_label_mask)
    native_scale = eye_crop_scale(compute_photo_to_eye_crop_affine(source_face.image_left_eye_xy, source_face.image_right_eye_xy, CROP_WIDTH, CROP_HEIGHT))
    for seed in range(10):
        target_scale = np.random.default_rng(seed).uniform(*ALWAYS_SIMULATE_CLEAN.crop_scale_range)
        photo, region_planes, left_eye, right_eye = dataset.simulate_phone_capture(
            source_face.photo_rgb, planes, source_face.image_left_eye_xy, source_face.image_right_eye_xy, np.random.default_rng(seed)
        )
        assert photo.shape[:2] == region_planes.shape[:2]
        crop_scale = eye_crop_scale(compute_photo_to_eye_crop_affine(left_eye, right_eye, CROP_WIDTH, CROP_HEIGHT))
        assert crop_scale == pytest.approx(min(target_scale, native_scale), rel=1e-3), seed


def test_capture_degradation_is_seeded_and_mild():
    rows, columns = np.mgrid[0:64, 0:64].astype(np.float32)
    photo_float = np.stack([0.5 + 0.3 * np.sin(columns / 7.0), 0.5 + 0.3 * np.cos(rows / 9.0), np.full_like(rows, 0.4)], axis=-1)
    photo = (photo_float * 255 + 0.5).astype(np.uint8)
    first = apply_capture_degradation(photo, ALWAYS_SIMULATE_DEGRADED, np.random.default_rng(5))
    second = apply_capture_degradation(photo, ALWAYS_SIMULATE_DEGRADED, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.uint8 and first.shape == photo.shape
    assert 0.0 < float(np.abs(first.astype(np.float32) - photo).mean()) < 8.0  # mild: a few levels
