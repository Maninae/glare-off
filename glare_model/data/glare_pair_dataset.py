"""GlarePairDataset: one synthetic (glare, clean, masks) training pair per item, made on the fly.

Per item: load a source face -> jitter the eye centers -> (a share of items) simulate a phone
capture: re-sample the photo region and labels to a phone-like crop scale, JPEG + noise -> cut the aligned
crop and the lens labels with the app's resampler -> photometric jitter + flip (training only) ->
call the glare synthesizer -> validate the contract -> return CHW float tensors.

- The crop uses `extract_eye_crop_with_area_prefilter` (bilinear, area pre-shrink below scale
  0.5), the exact resampling the app applies, so training never sees a cubic or unfiltered crop.
- Lens labels go through the SAME resampler as one-hot planes and are argmax'd back to labels,
  so the lens mask stays aligned with the image through every resize and prefilter.

Randomness:
- `fixed_seed` set (validation): item `i` uses `default_rng((fixed_seed, i))`, so every
  epoch sees the identical val set and metrics are comparable across epochs.
- `fixed_seed` None (training): the seed mixes `torch.initial_seed()` (differs per DataLoader
  worker and per epoch), the index, and a per-process draw counter, so samples never repeat
  but a run is reproducible from `GLOBAL.SEED`.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eye_crop.eye_crop_geometry import compute_photo_to_eye_crop_affine, extract_eye_crop_with_area_prefilter, eye_crop_scale
from glare_model.data.crop_augmentation import (
    EyeCenterJitterSettings,
    PhotometricJitterSettings,
    apply_photometric_jitter,
    flip_crop_and_lens_labels_horizontally,
    jitter_eye_centers,
)
from glare_model.data.phone_capture_simulation import (
    PhoneCaptureSimulationSettings,
    apply_capture_degradation,
    enlarge_crop_region_to_target_scale,
)
from glare_model.data.source_face import (
    LENS_LABEL_IMAGE_LEFT,
    LENS_LABEL_IMAGE_RIGHT,
    SYNTHESIS_IMAGE_KEYS,
    SYNTHESIS_MASK_KEYS,
    GlareSynthesizer,
    SourceFaceProvider,
    validate_synthesis_data_dict,
)

SEED_MODULUS = 2**63
UINT8_MAX = 255.0
LENS_LABEL_VALUES = (0, LENS_LABEL_IMAGE_LEFT, LENS_LABEL_IMAGE_RIGHT)


def lens_labels_to_one_hot_planes(lens_label_mask: np.ndarray) -> np.ndarray:
    """(H, W) uint8 labels -> (H, W, 3) float32 planes for background / image-left / image-right lens."""
    return np.stack([(lens_label_mask == label).astype(np.float32) for label in LENS_LABEL_VALUES], axis=-1)


def one_hot_planes_to_lens_labels(one_hot_planes: np.ndarray) -> np.ndarray:
    """Inverse of `lens_labels_to_one_hot_planes` after resampling: each pixel takes its majority label."""
    return np.asarray(LENS_LABEL_VALUES, dtype=np.uint8)[one_hot_planes.argmax(axis=-1)]


def cut_clean_crop_and_lens_labels(
    photo_rgb: np.ndarray,
    lens_label_planes: np.ndarray,
    image_left_eye_xy: np.ndarray,
    image_right_eye_xy: np.ndarray,
    crop_width: int,
    crop_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the clean crop and its lens labels through the same affine and the app's resampler.

    `photo_rgb` may be float [0, 1] or uint8 (the phone-capture path); the crop comes back float32 [0, 1].
    """
    photo_to_crop = compute_photo_to_eye_crop_affine(image_left_eye_xy, image_right_eye_xy, crop_width, crop_height)
    clean_eye_crop = extract_eye_crop_with_area_prefilter(photo_rgb, photo_to_crop, crop_width, crop_height)
    if clean_eye_crop.dtype == np.uint8:
        clean_eye_crop = clean_eye_crop.astype(np.float32) / UINT8_MAX
    label_planes_crop = extract_eye_crop_with_area_prefilter(lens_label_planes, photo_to_crop, crop_width, crop_height)
    return np.clip(clean_eye_crop, 0.0, 1.0).astype(np.float32), one_hot_planes_to_lens_labels(label_planes_crop)


@dataclass
class GlarePairDatasetSettings:
    """Per-split dataset behavior.

    Attributes:
        crop_height, crop_width: crop size handed to the synthesizer and the model.
        samples_per_source: items per source face per pass (each draws fresh jitter and glare).
        fixed_seed: None for training randomness, an int for a frozen validation set.
        use_photometric_augmentation: photometric jitter and flip (off for validation).

    Phone-capture simulation is NOT a per-split switch: it is passed to the dataset separately and
    applies to train and val alike (val draws are fixed by `fixed_seed`, so its mix is fixed).
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
        phone_capture_settings: PhoneCaptureSimulationSettings | None = None,
    ):
        self.source_face_provider = source_face_provider
        self.synthesize_glare = synthesize_glare
        self.dataset_settings = dataset_settings
        self.eye_jitter_settings = eye_jitter_settings
        self.photometric_settings = photometric_settings
        self.phone_capture_settings = phone_capture_settings
        self.draw_counter = itertools.count()

    def __len__(self) -> int:
        """Source faces times samples per source."""
        return len(self.source_face_provider) * self.dataset_settings.samples_per_source

    def make_random_generator(self, item_index: int) -> np.random.Generator:
        """Return the item's generator (fixed per index for validation; fresh per draw for training)."""
        if self.dataset_settings.fixed_seed is not None:
            return np.random.default_rng((self.dataset_settings.fixed_seed, item_index))
        return np.random.default_rng((torch.initial_seed() % SEED_MODULUS, item_index, next(self.draw_counter)))

    def should_simulate_phone_capture(self, random_generator: np.random.Generator) -> bool:
        """Draw whether this item is a simulated phone capture (no draw at all when disabled)."""
        capture_settings = self.phone_capture_settings
        if capture_settings is None or capture_settings.probability <= 0.0:
            return False
        return bool(random_generator.random() < capture_settings.probability)

    def simulate_phone_capture(
        self,
        photo_rgb: np.ndarray,
        lens_label_planes: np.ndarray,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        random_generator: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Re-sample the crop's photo region to a random phone-like crop scale, then JPEG + noise it.

        Returns the photo region, its label planes, and the eye centers in region pixels; cutting
        the crop from them goes through the area pre-shrink exactly as a phone photo does.
        """
        settings = self.dataset_settings
        target_crop_scale = random_generator.uniform(*self.phone_capture_settings.crop_scale_range)
        photo_rgb, lens_label_planes, left_eye, right_eye = enlarge_crop_region_to_target_scale(
            photo_rgb, lens_label_planes, left_eye, right_eye, settings.crop_width, settings.crop_height, target_crop_scale
        )
        photo_rgb = apply_capture_degradation(photo_rgb, self.phone_capture_settings, random_generator)
        return photo_rgb, lens_label_planes, left_eye, right_eye

    def __getitem__(self, item_index: int) -> dict[str, torch.Tensor | str]:
        """Build one pair; returns the Synthesis data_dict with (C, H, W) float32 tensors."""
        settings = self.dataset_settings
        random_generator = self.make_random_generator(item_index)
        source_face = self.source_face_provider.load_source_face(item_index // settings.samples_per_source)

        left_eye, right_eye = jitter_eye_centers(
            source_face.image_left_eye_xy, source_face.image_right_eye_xy, self.eye_jitter_settings, random_generator
        )
        photo_rgb = source_face.photo_rgb
        lens_label_planes = lens_labels_to_one_hot_planes(source_face.lens_label_mask)
        if self.should_simulate_phone_capture(random_generator):
            photo_rgb, lens_label_planes, left_eye, right_eye = self.simulate_phone_capture(
                photo_rgb, lens_label_planes, left_eye, right_eye, random_generator
            )
        clean_eye_crop, lens_label_mask = cut_clean_crop_and_lens_labels(
            photo_rgb, lens_label_planes, left_eye, right_eye, settings.crop_width, settings.crop_height
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
