"""Tests for glasses_classifier: net shapes, cache builder, dataset labels, metrics, ONNX export contract (CPU only)."""

import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from omegaconf import OmegaConf

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, extract_eye_crop_with_area_prefilter
from glasses_classifier.build_eye_crop_cache import CROP_HEIGHT, CROP_WIDTH, cache_crops_path, cache_rows_path, cut_crop_variants_for_row
from glasses_classifier.classification_metrics import (
    choose_threshold_for_glasses_recall,
    compute_binary_classification_report,
    compute_roc_auc,
)
from glasses_classifier.config_loading import load_glasses_classifier_config
from glasses_classifier.glasses_classifier_net import GlassesClassifierNet, build_glasses_classifier_from_config, count_trainable_parameters
from glasses_classifier.glasses_crop_dataset import GlassesCropDataset
from glasses_classifier.glasses_onnx_export import INPUT_NAME, OUTPUT_NAME, export_and_verify_glasses_classifier
from training_sources.load_source_manifest import SourceManifestRow

FAKE_FACE_LABELS = [True, True, True, False, True, False]
FAKE_VARIANT_COUNT = 3


@pytest.fixture
def fake_crop_cache(tmp_path: Path) -> Path:
    """A train split of 6 faces x 3 variants whose pixel value encodes (face, variant), plus its row index."""
    crops = np.lib.format.open_memmap(
        cache_crops_path(tmp_path, "train", FAKE_VARIANT_COUNT), mode="w+", dtype=np.uint8, shape=(len(FAKE_FACE_LABELS), FAKE_VARIANT_COUNT, CROP_HEIGHT, CROP_WIDTH, 3)
    )
    for face_index in range(len(FAKE_FACE_LABELS)):
        for variant_index in range(FAKE_VARIANT_COUNT):
            crops[face_index, variant_index] = 10 * face_index + variant_index
    crops.flush()
    del crops
    with open(cache_rows_path(tmp_path, "train"), "w") as rows_file:
        for face_index, has_glasses in enumerate(FAKE_FACE_LABELS):
            rows_file.write(json.dumps({"source_id": f"fake_{face_index}", "has_glasses": has_glasses, "photo_path": "unused"}) + "\n")
    return tmp_path


def test_net_outputs_one_probability_per_crop_and_fits_the_parameter_budget():
    """The configured net maps [B, 3, 256, 512] to [B, 1] in [0, 1] with 100k-400k parameters."""
    config = load_glasses_classifier_config()
    model = build_glasses_classifier_from_config(OmegaConf.to_container(config.MODEL)).eval()
    with torch.no_grad():
        probabilities = model(torch.rand(2, 3, CROP_HEIGHT, CROP_WIDTH))
    assert probabilities.shape == (2, 1)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert 100_000 <= count_trainable_parameters(model) <= 400_000


def test_eval_split_cache_crop_is_exactly_the_shared_app_crop(tmp_path: Path):
    """An evaluation-split crop (1 variant) equals extract_eye_crop_with_area_prefilter at the manifest eye centers."""
    photo_rgb = np.random.default_rng(0).integers(0, 256, size=(400, 600, 3), dtype=np.uint8)
    photo_path = tmp_path / "photo.png"
    cv2.imwrite(str(photo_path), cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR))
    left_eye, right_eye = np.array([250.0, 200.0]), np.array([350.0, 205.0])
    manifest_row = SourceManifestRow("fake", "test", photo_path, "CC0-1.0", "", left_eye, right_eye, tmp_path / "none.png", 0.0, has_glasses=False)
    crops = cut_crop_variants_for_row((manifest_row, 0, 1, 7, {}, {}))
    expected = extract_eye_crop_with_area_prefilter(photo_rgb, compute_photo_to_eye_crop_affine(left_eye, right_eye))
    assert crops.shape == (1, CROP_HEIGHT, CROP_WIDTH, 3)
    np.testing.assert_array_equal(crops[0], expected)


def test_train_cache_variants_are_jittered_and_deterministic(tmp_path: Path):
    """Train variants differ from each other and are reproducible from the same seed and row index."""
    rng = np.random.default_rng(1)
    photo_rgb = cv2.GaussianBlur(rng.integers(0, 256, size=(400, 600, 3), dtype=np.uint8), (0, 0), 3)
    photo_path = tmp_path / "photo.png"
    cv2.imwrite(str(photo_path), cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR))
    manifest_row = SourceManifestRow("fake", "train", photo_path, "CC0-1.0", "", np.array([250.0, 200.0]), np.array([350.0, 205.0]), tmp_path / "none.png", 0.0)
    first = cut_crop_variants_for_row((manifest_row, 3, 3, 7, {}, {}))
    second = cut_crop_variants_for_row((manifest_row, 3, 3, 7, {}, {}))
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert not np.array_equal(first[1], first[2])


def test_dataset_labels_follow_the_row_index_and_eval_reads_variant_zero(fake_crop_cache: Path):
    """Labels come from has_glasses; eval mode returns the untouched variant-0 crop."""
    dataset = GlassesCropDataset(fake_crop_cache, "train", is_training=False)
    for face_index, has_glasses in enumerate(FAKE_FACE_LABELS):
        sample = dataset[face_index]
        assert sample["has_glasses_label"].item() == float(has_glasses)
        assert sample["source_id"] == f"fake_{face_index}"
        assert sample["eye_crop"].shape == (3, CROP_HEIGHT, CROP_WIDTH)
        assert torch.allclose(sample["eye_crop"], torch.full_like(sample["eye_crop"], 10 * face_index / 255.0))


def test_training_samples_are_augmented_in_range_with_unchanged_labels(fake_crop_cache: Path):
    """Training mode keeps shape, range, and label while drawing different variants/augmentations."""
    dataset = GlassesCropDataset(fake_crop_cache, "train", is_training=True, grayscale_probability=0.5)
    samples = [dataset[3] for _ in range(6)]
    for sample in samples:
        assert sample["eye_crop"].shape == (3, CROP_HEIGHT, CROP_WIDTH)
        assert 0.0 <= sample["eye_crop"].min() and sample["eye_crop"].max() <= 1.0
        assert sample["has_glasses_label"].item() == 0.0
    assert len({round(float(sample["eye_crop"].mean()), 5) for sample in samples}) > 1


def test_class_balanced_weights_give_each_class_equal_total_mass(fake_crop_cache: Path):
    """Sum of weights over glasses faces equals the sum over bare faces."""
    dataset = GlassesCropDataset(fake_crop_cache, "train", is_training=True)
    weights = dataset.class_balanced_sample_weights().numpy()
    labels = np.array(FAKE_FACE_LABELS)
    assert weights[labels].sum() == pytest.approx(weights[~labels].sum())


def test_binary_report_counts_a_hand_checked_example():
    """2 glasses + 2 bare faces, one of each misclassified at threshold 0.5."""
    report = compute_binary_classification_report(np.array([0.9, 0.3, 0.6, 0.1]), np.array([1, 1, 0, 0]), 0.5)
    assert report.confusion_matrix_bare_glasses == [[1, 1], [1, 1]]
    assert report.accuracy == 0.5
    assert report.glasses_recall == 0.5 and report.glasses_precision == 0.5
    assert report.bare_recall == 0.5 and report.bare_precision == 0.5


def test_roc_auc_on_separable_inverted_and_tied_scores():
    """Perfect ranking is 1, inverted is 0, all-tied is 0.5."""
    labels = np.array([0, 0, 1, 1])
    assert compute_roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert compute_roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert compute_roc_auc(np.full(4, 0.5), labels) == 0.5


def test_threshold_stays_at_default_when_it_meets_the_recall_floor():
    """A clean gap around 0.5 keeps the default."""
    probabilities = np.array([0.8, 0.9, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert choose_threshold_for_glasses_recall(probabilities, labels, min_glasses_recall=1.0, default_threshold=0.5) == 0.5


def test_threshold_is_lowered_only_as_far_as_the_recall_floor_needs():
    """A glasses face at 0.42 forces the threshold down to 0.4 (the highest grid value keeping recall 1.0)."""
    probabilities = np.array([0.42, 0.9, 0.1, 0.2])
    labels = np.array([1, 1, 0, 0])
    assert choose_threshold_for_glasses_recall(probabilities, labels, min_glasses_recall=1.0, default_threshold=0.5) == pytest.approx(0.4)


def test_export_meets_the_app_contract(tmp_path: Path):
    """Opset 17, input eye_crop [1,3,256,512], output glasses_probability [1,1], parity with PyTorch, small file."""
    torch.manual_seed(0)
    model = GlassesClassifierNet().eval()
    output_path = tmp_path / "glasses_classifier.onnx"
    export_report = export_and_verify_glasses_classifier(model, output_path)
    assert export_report["webgpu_operator_check"] == "pass"
    assert export_report["opset"] == 17
    assert export_report["file_size_mb"] < 2.0

    onnx_model = onnx.load(str(output_path))
    input_dims = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    output_dims = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    assert onnx_model.graph.input[0].name == INPUT_NAME and input_dims == [1, 3, CROP_HEIGHT, CROP_WIDTH]
    assert onnx_model.graph.output[0].name == OUTPUT_NAME and output_dims == [1, 1]

    eye_crop = np.random.default_rng(3).random((1, 3, CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    (onnx_probability,) = session.run([OUTPUT_NAME], {INPUT_NAME: eye_crop})
    with torch.no_grad():
        torch_probability = model(torch.from_numpy(eye_crop)).numpy()
    np.testing.assert_allclose(onnx_probability, torch_probability, atol=2e-3)
