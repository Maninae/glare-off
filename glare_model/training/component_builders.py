"""Build the model, datasets, and DataLoaders from a composed training config.

Importing this module imports every module that registers into a registry (the architecture,
the fake data sources, the real-module adapters), so a config can name any of them.
"""

import logging
from typing import Any

import cv2

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import glare_model.architecture.glare_removal_nafnet  # noqa: F401  (registers GlareRemovalNAFNet)
import glare_model.data.fake_data_sources  # noqa: F401  (registers the FAKE provider and synthesizer)
import glare_model.data.real_module_adapters  # noqa: F401  (registers the real-module adapters)
from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet, count_trainable_parameters
from glare_model.data.crop_augmentation import EyeCenterJitterSettings, PhotometricJitterSettings
from glare_model.data.glare_pair_dataset import GlarePairDataset, GlarePairDatasetSettings
from glare_model.data.phone_capture_simulation import PhoneCaptureSimulationSettings
from glare_model.losses.glare_removal_loss import GlareRemovalLoss, GlareRemovalLossWeights
from glare_model.registry import (
    GLARE_SYNTHESIZER_REGISTRY,
    MODEL_REGISTRY,
    SOURCE_FACE_PROVIDER_REGISTRY,
    build_from_config_section,
)

logger = logging.getLogger(__name__)

VALIDATION_SPLIT = "val"
TRAINING_SPLIT = "train"


def config_section_as_dict(config_section: DictConfig) -> dict[str, Any]:
    """Return a plain, resolved python dict of a config section."""
    return OmegaConf.to_container(config_section, resolve=True)


def lowercase_section_kwargs(config_section: DictConfig) -> dict[str, Any]:
    """`{"BASE_LR": 1}` -> `{"base_lr": 1}` for dataclass construction."""
    return {key.lower(): value for key, value in config_section_as_dict(config_section).items()}


def build_glare_model(model_config: DictConfig | dict[str, Any]) -> GlareRemovalNAFNet:
    """Build the registered model named in `MODEL.NAME` and log its parameter count."""
    model_section = model_config if isinstance(model_config, dict) else config_section_as_dict(model_config)
    model = build_from_config_section(MODEL_REGISTRY, model_section)
    logger.info("%s: %.3fM trainable parameters", model_section["NAME"], count_trainable_parameters(model) / 1e6)
    return model


def build_glare_removal_loss(loss_config: DictConfig) -> GlareRemovalLoss:
    """Build the composite loss from `LOSS.WEIGHTS`."""
    return GlareRemovalLoss(GlareRemovalLossWeights(**lowercase_section_kwargs(loss_config.WEIGHTS)))


def build_glare_pair_dataset(training_config: DictConfig, split: str) -> GlarePairDataset:
    """Build the train or val dataset; val uses a fixed seed and no photometric augmentation.

    Phone-capture simulation (`DATA.PHONE_CAPTURE`) applies to both splits; val's draws are seeded,
    so its share of phone-scale samples is fixed across evaluations.
    """
    data_config = training_config.DATA
    is_validation = split == VALIDATION_SPLIT
    source_face_provider = build_from_config_section(
        SOURCE_FACE_PROVIDER_REGISTRY, config_section_as_dict(data_config.SOURCE_FACE_PROVIDER[split.upper()]), split=split
    )
    synthesize_glare = build_from_config_section(GLARE_SYNTHESIZER_REGISTRY, config_section_as_dict(data_config.GLARE_SYNTHESIZER))
    dataset_settings = GlarePairDatasetSettings(
        crop_height=data_config.CROP_HEIGHT,
        crop_width=data_config.CROP_WIDTH,
        samples_per_source=data_config.VAL_SAMPLES_PER_SOURCE if is_validation else 1,
        fixed_seed=data_config.VAL_SEED if is_validation else None,
        use_photometric_augmentation=not is_validation,
    )
    return GlarePairDataset(
        source_face_provider,
        synthesize_glare,
        dataset_settings,
        EyeCenterJitterSettings(**lowercase_section_kwargs(data_config.EYE_JITTER)),
        PhotometricJitterSettings(**lowercase_section_kwargs(data_config.PHOTOMETRIC)),
        PhoneCaptureSimulationSettings(**lowercase_section_kwargs(data_config.PHONE_CAPTURE)),
    )


def limit_opencv_threads_in_worker(worker_id: int) -> None:
    """DataLoader worker init: one OpenCV thread per worker so workers do not oversubscribe the CPU."""
    cv2.setNumThreads(1)


def build_data_loader(dataset: GlarePairDataset, training_config: DictConfig, shuffle: bool) -> DataLoader:
    """Training loader when `shuffle` (drops the ragged last batch), else the val loader."""
    worker_count = training_config.DATALOADER.NUM_WORKERS
    batch_size = training_config.DATALOADER.BATCH_SIZE if shuffle else training_config.DATALOADER.VAL_BATCH_SIZE
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=worker_count,
        persistent_workers=worker_count > 0,
        drop_last=shuffle,
        pin_memory=False,
        worker_init_fn=limit_opencv_threads_in_worker,
    )
