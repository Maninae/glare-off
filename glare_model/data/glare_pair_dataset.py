"""GlarePairDataset: one synthetic (glare, clean, masks) training pair per item, made on the fly.

Per item: load a source face -> jitter the eye centers -> cut the aligned crop and the lens
labels with `eye_crop.eye_crop_geometry` -> photometric jitter + flip (training only) -> call the
glare synthesizer -> validate the contract -> return CHW float tensors.

Randomness:
- `fixed_seed` set (validation): item `i` uses `default_rng((fixed_seed, i))`, so every
  epoch sees the identical val set and metrics are comparable across epochs.
- `fixed_seed` None (training): the seed mixes `torch.initial_seed()` (differs per DataLoader
  worker and per epoch), the index, and a per-process draw counter, so samples never repeat
  but a run is reproducible from `GLOBAL.SEED`.
"""

import itertools
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, extract_eye_crop, eye_crop_scale
from glare_model.data.crop_augmentation import (
    EyeCenterJitterSettings,
    PhotometricJitterSettings,
    apply_photometric_jitter,
    flip_crop_and_lens_labels_horizontally,
    jitter_eye_centers,
)
from glare_model.data.source_face import (
    SYNTHESIS_IMAGE_KEYS,
    SYNTHESIS_MASK_KEYS,
    GlareSynthesizer,
    SourceFaceProvider,
    validate_synthesis_data_dict,
)

SEED_MODULUS = 2**63


@dataclass
class GlarePairDatasetSettings:
    """Per-split dataset behavior.

    Attributes:
        crop_height, crop_width: crop size handed to the synthesizer and the model.
        samples_per_source: items per source face per pass (each draws fresh jitter and glare).
        fixed_seed: None for training randomness, an int for a frozen validation set.
        use_photometric_augmentation: photometric jitter and flip (off for validation).
    """

    crop_height: int = 256
    crop_width: int = 512
    samples_per_source: int = 1
    fixed_seed: int | None = None
    use_photometric_augmentation: bool = True


class GlarePairDataset(Dataset):
    """Map-style dataset of synthetic glare pairs built from a source-face provider and a synthesizer."""

    def __init__(
        self,
        source_face_provider: SourceFaceProvider,
        synthesize_glare: GlareSynthesizer,
        dataset_settings: GlarePairDatasetSettings,
        eye_jitter_settings: EyeCenterJitterSettings,
        photometric_settings: PhotometricJitterSettings,
    ):
        self.source_face_provider = source_face_provider
        self.synthesize_glare = synthesize_glare
        self.dataset_settings = dataset_settings
        self.eye_jitter_settings = eye_jitter_settings
        self.photometric_settings = photometric_settings
        self.draw_counter = itertools.count()

    def __len__(self) -> int:
        """Source faces times samples per source."""
        return len(self.source_face_provider) * self.dataset_settings.samples_per_source

    def make_random_generator(self, item_index: int) -> np.random.Generator:
        """Return the item's generator (fixed per index for validation; fresh per draw for training)."""
        if self.dataset_settings.fixed_seed is not None:
            return np.random.default_rng((self.dataset_settings.fixed_seed, item_index))
        return np.random.default_rng((torch.initial_seed() % SEED_MODULUS, item_index, next(self.draw_counter)))

    def __getitem__(self, item_index: int) -> dict[str, torch.Tensor | str]:
        """Build one pair; returns the Synthesis data_dict with (C, H, W) float32 tensors."""
        settings = self.dataset_settings
        random_generator = self.make_random_generator(item_index)
        source_face = self.source_face_provider.load_source_face(item_index // settings.samples_per_source)

        left_eye, right_eye = jitter_eye_centers(
            source_face.image_left_eye_xy, source_face.image_right_eye_xy, self.eye_jitter_settings, random_generator
        )
        photo_to_crop = compute_photo_to_eye_crop_affine(left_eye, right_eye, settings.crop_width, settings.crop_height)
        interpolation = cv2.INTER_AREA if eye_crop_scale(photo_to_crop) < 1.0 else cv2.INTER_CUBIC
        clean_eye_crop = extract_eye_crop(
            source_face.photo_rgb, photo_to_crop, settings.crop_width, settings.crop_height, interpolation
        )
        clean_eye_crop = np.clip(clean_eye_crop, 0.0, 1.0).astype(np.float32)
        lens_label_mask = extract_eye_crop(
            source_face.lens_label_mask, photo_to_crop, settings.crop_width, settings.crop_height, cv2.INTER_NEAREST
        )

        if settings.use_photometric_augmentation:
            clean_eye_crop = apply_photometric_jitter(clean_eye_crop, self.photometric_settings, random_generator)
            if random_generator.random() < self.photometric_settings.horizontal_flip_probability:
                clean_eye_crop, lens_label_mask = flip_crop_and_lens_labels_horizontally(clean_eye_crop, lens_label_mask)

        synthesis_data_dict = self.synthesize_glare(clean_eye_crop, lens_label_mask, random_generator)
        validate_synthesis_data_dict(synthesis_data_dict, settings.crop_height, settings.crop_width)

        sample_data_dict: dict[str, torch.Tensor | str] = {
            key: torch.from_numpy(np.ascontiguousarray(synthesis_data_dict[key].transpose(2, 0, 1)))
            for key in SYNTHESIS_IMAGE_KEYS + SYNTHESIS_MASK_KEYS
        }
        sample_data_dict["source_id"] = source_face.source_id
        return sample_data_dict
