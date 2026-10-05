"""Shared fixtures for the glare_model tests: a tiny model config and small fake datasets (CPU only)."""

import pytest
import torch

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet
from glare_model.data.crop_augmentation import EyeCenterJitterSettings, PhotometricJitterSettings
from glare_model.data.fake_data_sources import FakeSourceFaceProvider, synthesize_fake_blob_glare
from glare_model.data.glare_pair_dataset import GlarePairDataset, GlarePairDatasetSettings

TINY_MODEL_KWARGS = {
    "base_width": 8,
    "encoder_block_counts": [1, 1, 1, 1],
    "middle_block_count": 1,
    "decoder_block_counts": [1, 1, 1, 1],
}
SMALL_CROP_HEIGHT = 64
SMALL_CROP_WIDTH = 128


@pytest.fixture
def tiny_glare_model() -> GlareRemovalNAFNet:
    """A width-8 model with non-trivial weights (residual scales and heads randomized)."""
    model = GlareRemovalNAFNet(**TINY_MODEL_KWARGS)
    torch.manual_seed(0)
    for parameter_name, parameter in model.named_parameters():
        if "residual_scale" in parameter_name or "head" in parameter_name:
            parameter.data.normal_(0.0, 0.2)
    return model.eval()


@pytest.fixture
def fake_glare_pair_dataset() -> GlarePairDataset:
    """A training-mode fake dataset of 8 faces at 64x128."""
    return GlarePairDataset(
        FakeSourceFaceProvider("train", face_count=8),
        synthesize_fake_blob_glare,
        GlarePairDatasetSettings(crop_height=SMALL_CROP_HEIGHT, crop_width=SMALL_CROP_WIDTH),
        EyeCenterJitterSettings(),
        PhotometricJitterSettings(),
    )


@pytest.fixture
def tiny_model_kwargs() -> dict:
    """Constructor kwargs of the tiny test model (also used as a MODEL config section)."""
    return dict(TINY_MODEL_KWARGS)
