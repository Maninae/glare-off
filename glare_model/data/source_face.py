"""The two narrow interfaces the training dataset depends on, so data sources plug in by config.

- A source-face provider is indexable: `len(provider)` faces, `provider.load_source_face(i)`
  returns one `SourceFace` (a full photo plus its eye centers and lens label mask). Real one:
  `real_module_adapters.ManifestSourceFaceProvider`; fake one: `fake_data_sources.FakeSourceFaceProvider`.
- A glare synthesizer is a callable `synthesize(clean_eye_crop, lens_label_mask, rng)` that
  returns the "Synthesis data_dict" from the repo CLAUDE.md contract. Real one:
  `real_module_adapters.build_glare_synthesis_module_synthesizer`; fake one:
  `fake_data_sources.synthesize_fake_blob_glare`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

# Lens label values on the source photo's pixel grid (source manifest contract).
LENS_LABEL_BACKGROUND = 0
LENS_LABEL_IMAGE_LEFT = 1
LENS_LABEL_IMAGE_RIGHT = 2

SYNTHESIS_IMAGE_KEYS = ("clean_eye_crop", "glare_eye_crop")
SYNTHESIS_MASK_KEYS = ("glare_mask", "lost_detail_mask", "lens_mask")


@dataclass
class SourceFace:
    """One clean source face: the full photo plus everything needed to cut its eye crop.

    Attributes:
        source_id: manifest id, carried into the data_dict for debugging and per-source metrics.
        photo_rgb: (H, W, 3) float32 sRGB in [0, 1].
        image_left_eye_xy: (x, y) of the eye on the image's left side, photo pixels.
        image_right_eye_xy: (x, y) of the eye on the image's right side, photo pixels.
        lens_label_mask: (H, W) uint8, 0 background / 1 image-left lens / 2 image-right lens.
    """

    source_id: str
    photo_rgb: np.ndarray
    image_left_eye_xy: np.ndarray
    image_right_eye_xy: np.ndarray
    lens_label_mask: np.ndarray


class SourceFaceProvider(Protocol):
    """Indexable collection of `SourceFace`s for one split."""

    def __len__(self) -> int:
        """Number of source faces in this split."""

    def load_source_face(self, source_index: int) -> SourceFace:
        """Load face number `source_index` (0 <= index < len)."""


GlareSynthesizer = Callable[[np.ndarray, np.ndarray, np.random.Generator], dict[str, Any]]


def validate_synthesis_data_dict(synthesis_data_dict: dict[str, Any], crop_height: int, crop_width: int) -> None:
    """Fail loudly if a synthesizer's output breaks the Synthesis data_dict contract.

    Checked at the dataset boundary so a contract drift in `glare_synthesis/` surfaces as a clear
    error on the first sample, not as a silent shape broadcast deep inside the loss.
    """
    expected_channels = {key: 3 for key in SYNTHESIS_IMAGE_KEYS} | {key: 1 for key in SYNTHESIS_MASK_KEYS}
    for key, channel_count in expected_channels.items():
        if key not in synthesis_data_dict:
            raise KeyError(f"synthesis data_dict is missing {key!r}; has {sorted(synthesis_data_dict)}")
        array = np.asarray(synthesis_data_dict[key])
        expected_shape = (crop_height, crop_width, channel_count)
        if array.shape != expected_shape:
            raise ValueError(f"synthesis {key!r} has shape {array.shape}, expected {expected_shape}")
        if array.dtype != np.float32:
            raise ValueError(f"synthesis {key!r} has dtype {array.dtype}, expected float32")
