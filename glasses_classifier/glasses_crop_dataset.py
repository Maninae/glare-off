"""GlassesCropDataset: cached app-exact eye crops -> {"eye_crop", "has_glasses_label", ...} data_dicts.

Reads the uint8 memmaps written by `build_eye_crop_cache` (one file per split, (N, K, 256, 512, 3)).

- Training: each access picks one of the K cached jitter variants at random, then applies glare_model's
  photometric jitter, an occasional grayscale, and a horizontal flip (no label to swap: a mirrored face
  still does or does not wear glasses).
- Evaluation: variant 0 (the exact manifest eye centers), no augmentation.
- `class_balanced_sample_weights` gives each face 1 / (count of its class), for a WeightedRandomSampler
  that draws glasses and bare faces equally often.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from glare_model.data.crop_augmentation import PhotometricJitterSettings, apply_photometric_jitter
from glasses_classifier.build_eye_crop_cache import cache_rows_path

UINT8_MAX = 255.0


def find_split_crops_path(cache_directory: Path, split: str) -> Path:
    """Return the one crop memmap of `split` (its variant count is in the filename)."""
    candidates = sorted(cache_directory.glob(f"{split}__eye_crops__variants=*__*.npy"))
    candidates = [path for path in candidates if not path.name.endswith(".partial.npy")]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one finished crop cache for split {split!r} in {cache_directory}, found {candidates}")
    return candidates[0]


class GlassesCropDataset(Dataset):
    """One cached split of eye crops with their has_glasses labels.

    Args:
        cache_directory: `CACHE.DIRECTORY`.
        split: train / val / test / real_glare_eval.
        is_training: random variant + augmentation when True; variant 0, untouched, when False.
        photometric_settings: jitter ranges and flip probability (training only).
        grayscale_probability: share of training crops turned to gray (training only).
    """

    def __init__(
        self,
        cache_directory: Path,
        split: str,
        is_training: bool,
        photometric_settings: PhotometricJitterSettings | None = None,
        grayscale_probability: float = 0.0,
    ):
        self.split = split
        self.is_training = is_training
        self.photometric_settings = photometric_settings or PhotometricJitterSettings()
        self.grayscale_probability = grayscale_probability
        self.crops_path = find_split_crops_path(Path(cache_directory), split)
        with open(cache_rows_path(Path(cache_directory), split)) as rows_file:
            self.face_rows = [json.loads(line) for line in rows_file if line.strip()]
        self.has_glasses_labels = np.array([row["has_glasses"] for row in self.face_rows], dtype=np.float32)
        # Opened lazily per process: a memmap handle must not be pickled into DataLoader workers.
        self.crop_memmap: np.ndarray | None = None
        crop_header = np.load(self.crops_path, mmap_mode="r")
        if crop_header.shape[0] != len(self.face_rows):
            raise ValueError(f"{self.crops_path} has {crop_header.shape[0]} faces but the row index has {len(self.face_rows)}")
        self.variant_count = crop_header.shape[1]

    def __len__(self) -> int:
        return len(self.face_rows)

    def read_crop_uint8(self, index: int, variant_index: int = 0) -> np.ndarray:
        """Return one cached (256, 512, 3) uint8 RGB crop."""
        if self.crop_memmap is None:
            self.crop_memmap = np.load(self.crops_path, mmap_mode="r")
        return np.asarray(self.crop_memmap[index, variant_index])

    def augment_crop(self, eye_crop: np.ndarray, random_generator: np.random.Generator) -> np.ndarray:
        """Photometric jitter, optional grayscale, optional flip; float32 [0, 1] HWC in and out."""
        eye_crop = apply_photometric_jitter(eye_crop, self.photometric_settings, random_generator)
        if random_generator.random() < self.grayscale_probability:
            eye_crop = np.repeat(eye_crop.mean(axis=2, keepdims=True), 3, axis=2)
        if random_generator.random() < self.photometric_settings.horizontal_flip_probability:
            eye_crop = eye_crop[:, ::-1]
        return np.ascontiguousarray(eye_crop, dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        """Return {"eye_crop": [3, 256, 512] float32 in [0, 1], "has_glasses_label": [1] float32, "source_id", "face_index"}."""
        if self.is_training:
            random_generator = np.random.default_rng([torch.initial_seed() % 2**32, index, np.random.randint(2**31)])
            variant_index = int(random_generator.integers(self.variant_count))
            eye_crop = self.augment_crop(self.read_crop_uint8(index, variant_index).astype(np.float32) / UINT8_MAX, random_generator)
        else:
            eye_crop = self.read_crop_uint8(index, 0).astype(np.float32) / UINT8_MAX
        return {
            "eye_crop": torch.from_numpy(eye_crop.transpose(2, 0, 1).copy()),
            "has_glasses_label": torch.tensor([self.has_glasses_labels[index]]),
            "source_id": self.face_rows[index]["source_id"],
            "face_index": index,
        }

    def class_balanced_sample_weights(self) -> torch.Tensor:
        """Per-face sampling weight 1 / count(face's class), so both classes are drawn equally often."""
        glasses_count = float(self.has_glasses_labels.sum())
        bare_count = float(len(self.has_glasses_labels) - glasses_count)
        if glasses_count == 0 or bare_count == 0:
            raise ValueError(f"split {self.split} has only one class; cannot balance")
        weights = np.where(self.has_glasses_labels == 1.0, 1.0 / glasses_count, 1.0 / bare_count)
        return torch.from_numpy(weights.astype(np.float64))
