"""Dataset contract with the FAKE sources, determinism rules, flips, and the real-module adapters."""

import json

import cv2
import numpy as np
import pytest
import torch

from glare_model.data.crop_augmentation import (
    EyeCenterJitterSettings,
    PhotometricJitterSettings,
    flip_crop_and_lens_labels_horizontally,
)
from glare_model.data.fake_data_sources import FakeSourceFaceProvider, synthesize_fake_blob_glare
from glare_model.data.glare_pair_dataset import GlarePairDataset, GlarePairDatasetSettings
from glare_model.data.real_module_adapters import ManifestSourceFaceProvider, build_glare_synthesis_module_synthesizer
from glare_model.data.source_face import validate_synthesis_data_dict

EXPECTED_CHANNELS = {"clean_eye_crop": 3, "glare_eye_crop": 3, "glare_mask": 1, "lost_detail_mask": 1, "lens_mask": 1}


def assert_sample_honors_contract(sample_data_dict: dict, height: int, width: int) -> None:
    """Every contract key, as a (C, H, W) float32 tensor in [0, 1], plus a string source_id."""
    for key, channel_count in EXPECTED_CHANNELS.items():
        tensor = sample_data_dict[key]
        assert tensor.shape == (channel_count, height, width), key
        assert tensor.dtype == torch.float32, key
        assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0, key
    assert isinstance(sample_data_dict["source_id"], str) and sample_data_dict["source_id"]


def test_fake_dataset_items_honor_the_synthesis_contract(fake_glare_pair_dataset):
    for item_index in range(len(fake_glare_pair_dataset)):
        assert_sample_honors_contract(fake_glare_pair_dataset[item_index], 64, 128)


def test_glare_only_changes_pixels_inside_the_lens(fake_glare_pair_dataset):
    for item_index in range(len(fake_glare_pair_dataset)):
        sample = fake_glare_pair_dataset[item_index]
        outside_lens = sample["lens_mask"] == 0
        change = (sample["glare_eye_crop"] - sample["clean_eye_crop"]).abs()
        assert float(change[outside_lens.expand_as(change)].max()) == 0.0


def test_fixed_seed_dataset_is_identical_across_reads():
    validation_dataset = GlarePairDataset(
        FakeSourceFaceProvider("val", face_count=4),
        synthesize_fake_blob_glare,
        GlarePairDatasetSettings(crop_height=64, crop_width=128, fixed_seed=7, use_photometric_augmentation=False),
        EyeCenterJitterSettings(),
        PhotometricJitterSettings(),
    )
    first_read, second_read = validation_dataset[3], validation_dataset[3]
    for key in EXPECTED_CHANNELS:
        assert torch.equal(first_read[key], second_read[key]), key


def test_training_dataset_draws_fresh_samples(fake_glare_pair_dataset):
    first_read, second_read = fake_glare_pair_dataset[0], fake_glare_pair_dataset[0]
    assert not torch.equal(first_read["clean_eye_crop"], second_read["clean_eye_crop"])


def test_flip_swaps_lens_labels():
    clean_eye_crop = np.zeros((4, 6, 3), dtype=np.float32)
    lens_labels = np.zeros((4, 6), dtype=np.uint8)
    lens_labels[:, 0] = 1
    lens_labels[:, 5] = 2
    _, flipped_labels = flip_crop_and_lens_labels_horizontally(clean_eye_crop, lens_labels)
    assert (flipped_labels[:, 0] == 1).all() and (flipped_labels[:, 5] == 2).all()
    assert (lens_labels[:, 0] == 1).all()  # input untouched


def test_contract_validation_rejects_a_wrong_mask_shape():
    bad_output = synthesize_fake_blob_glare(np.zeros((32, 64, 3), np.float32), np.zeros((32, 64), np.uint8), np.random.default_rng(0))
    bad_output["glare_mask"] = bad_output["glare_mask"][..., 0]
    with pytest.raises(ValueError, match="glare_mask"):
        validate_synthesis_data_dict(bad_output, 32, 64)


def write_temporary_manifest(manifest_directory, face_count: int = 2) -> str:
    """Write fake faces as PNGs plus a contract-shaped manifest JSONL; return the manifest path."""
    fake_provider = FakeSourceFaceProvider("train", face_count=face_count)
    manifest_lines = []
    for face_index in range(face_count):
        source_face = fake_provider.load_source_face(face_index)
        photo_path = manifest_directory / f"photo_{face_index}.png"
        mask_path = manifest_directory / f"lens_{face_index}.png"
        cv2.imwrite(str(photo_path), cv2.cvtColor((source_face.photo_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), source_face.lens_label_mask)
        manifest_lines.append(json.dumps({
            "source_id": source_face.source_id, "split": "train", "photo_path": str(photo_path),
            "license": "test", "attribution": "test", "image_left_eye_xy": source_face.image_left_eye_xy.tolist(),
            "image_right_eye_xy": source_face.image_right_eye_xy.tolist(), "lens_mask_path": str(mask_path),
            "source_glare_score": 0.0,
        }))
    manifest_path = manifest_directory / "source_manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    return str(manifest_path)


def test_real_modules_produce_contract_samples_through_the_adapters(tmp_path):
    """Integration: real `training_sources` manifest loader + real `glare_synthesis` renderer."""
    provider = ManifestSourceFaceProvider(split="train", manifest_path=write_temporary_manifest(tmp_path))
    assert len(provider) == 2
    real_dataset = GlarePairDataset(
        provider,
        build_glare_synthesis_module_synthesizer(),
        GlarePairDatasetSettings(crop_height=128, crop_width=256, fixed_seed=3),
        EyeCenterJitterSettings(),
        PhotometricJitterSettings(),
    )
    for item_index in range(len(real_dataset)):
        sample = real_dataset[item_index]
        assert_sample_honors_contract(sample, 128, 256)
        assert sample["source_id"].startswith("fake_train_")
