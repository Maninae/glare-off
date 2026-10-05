"""FAKE source faces and FAKE glare, for tests and smoke runs only. Nothing here resembles real data.

- `FakeSourceFaceProvider`: procedurally drawn "photos": smooth random color fields with two
  cartoon eyes (sclera, iris, pupil), and, for most faces, elliptical lenses with a dark frame
  and the matching 0/1/2 lens label mask. Face `i` of a split is deterministic.
- `synthesize_fake_blob_glare`: adds colored Gaussian and disc blobs inside the lenses, clips at
  1, and derives the glare and lost-detail masks the way the contract defines them.

The point is to exercise every code path (contract shapes, flips, empty lenses, saturation) and
to prove the network can learn to undo an additive in-lens layer, not to model real glare.
"""

import cv2
import numpy as np

from glare_model.data.source_face import LENS_LABEL_IMAGE_LEFT, LENS_LABEL_IMAGE_RIGHT, GlareSynthesizer, SourceFace
from glare_model.registry import GLARE_SYNTHESIZER_REGISTRY, SOURCE_FACE_PROVIDER_REGISTRY

FAKE_PHOTO_SIZE = 512
FAKE_EYE_DISTANCE_RANGE = (90.0, 150.0)
FAKE_ROLL_DEGREES = 10.0
FAKE_GLASSES_PROBABILITY = 0.8
FAKE_LENS_SEMI_AXES_FRACTION = (0.42, 0.3)  # of eye distance: (half width, half height)
FAKE_FRAME_THICKNESS = 4
SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}

FAKE_NO_GLARE_PROBABILITY = 0.15
FAKE_GLARE_TINTS_RGB = (
    (1.0, 1.0, 1.0),
    (0.35, 1.0, 0.45),
    (0.4, 0.6, 1.0),
    (0.8, 0.45, 1.0),
)
# A pixel counts as glare once any channel moved by this much; masks ramp linearly up to it.
GLARE_MASK_FULL_CHANGE = 0.08
LOST_DETAIL_SATURATION_LEVEL = 0.97
LOST_DETAIL_MIN_CHANGE = 0.25


def draw_smooth_random_field(height: int, width: int, random_generator: np.random.Generator) -> np.ndarray:
    """Return an (H, W, 3) float32 image of low-frequency color plus faint mid-frequency texture."""
    coarse = random_generator.uniform(0.2, 0.8, size=(6, 6, 3)).astype(np.float32)
    texture = random_generator.normal(0.0, 0.04, size=(48, 48, 3)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    field += cv2.resize(texture, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(field, 0.0, 1.0)


@SOURCE_FACE_PROVIDER_REGISTRY.register
class FakeSourceFaceProvider:
    """FAKE provider: `face_count` deterministic procedural faces for one split."""

    def __init__(self, split: str, face_count: int = 256):
        self.split = split
        self.face_count = face_count

    def __len__(self) -> int:
        """Number of fake faces in this split."""
        return self.face_count

    def load_source_face(self, source_index: int) -> SourceFace:
        """Draw fake face `source_index`; the same index always returns the same face."""
        random_generator = np.random.default_rng(SPLIT_SEED_OFFSETS[self.split] + source_index)
        photo_rgb = draw_smooth_random_field(FAKE_PHOTO_SIZE, FAKE_PHOTO_SIZE, random_generator)
        lens_label_mask = np.zeros((FAKE_PHOTO_SIZE, FAKE_PHOTO_SIZE), dtype=np.uint8)

        eye_distance = random_generator.uniform(*FAKE_EYE_DISTANCE_RANGE)
        roll_radians = np.radians(random_generator.uniform(-FAKE_ROLL_DEGREES, FAKE_ROLL_DEGREES))
        midpoint = FAKE_PHOTO_SIZE / 2 + random_generator.uniform(-40, 40, size=2)
        half_offset = 0.5 * eye_distance * np.array([np.cos(roll_radians), np.sin(roll_radians)])
        left_eye, right_eye = midpoint - half_offset, midpoint + half_offset

        has_glasses = random_generator.random() < FAKE_GLASSES_PROBABILITY
        for eye_center, lens_label in ((left_eye, LENS_LABEL_IMAGE_LEFT), (right_eye, LENS_LABEL_IMAGE_RIGHT)):
            draw_fake_eye(photo_rgb, eye_center, eye_distance, np.degrees(roll_radians), random_generator)
            if has_glasses:
                draw_fake_lens(photo_rgb, lens_label_mask, eye_center, eye_distance, np.degrees(roll_radians), lens_label)

        return SourceFace(
            source_id=f"fake_{self.split}_{source_index:06d}",
            photo_rgb=photo_rgb,
            image_left_eye_xy=left_eye,
            image_right_eye_xy=right_eye,
            lens_label_mask=lens_label_mask,
        )


def draw_fake_eye(
    photo_rgb: np.ndarray, eye_center: np.ndarray, eye_distance: float, roll_degrees: float, random_generator: np.random.Generator
) -> None:
    """In place: draw a sclera ellipse, a colored iris, and a pupil at `eye_center`."""
    center = tuple(int(round(value)) for value in eye_center)
    sclera_axes = (int(0.18 * eye_distance), int(0.08 * eye_distance))
    cv2.ellipse(photo_rgb, center, sclera_axes, roll_degrees, 0, 360, (0.92, 0.9, 0.88), -1, cv2.LINE_AA)
    iris_color = tuple(float(value) for value in random_generator.uniform(0.1, 0.6, size=3))
    cv2.circle(photo_rgb, center, int(0.075 * eye_distance), iris_color, -1, cv2.LINE_AA)
    cv2.circle(photo_rgb, center, int(0.03 * eye_distance), (0.03, 0.03, 0.03), -1, cv2.LINE_AA)


def draw_fake_lens(
    photo_rgb: np.ndarray,
    lens_label_mask: np.ndarray,
    eye_center: np.ndarray,
    eye_distance: float,
    roll_degrees: float,
    lens_label: int,
) -> None:
    """In place: fill the lens ellipse in the label mask and draw a dark frame around it."""
    center = tuple(int(round(value)) for value in eye_center)
    lens_axes = (int(FAKE_LENS_SEMI_AXES_FRACTION[0] * eye_distance), int(FAKE_LENS_SEMI_AXES_FRACTION[1] * eye_distance))
    cv2.ellipse(lens_label_mask, center, lens_axes, roll_degrees, 0, 360, int(lens_label), -1)
    cv2.ellipse(photo_rgb, center, lens_axes, roll_degrees, 0, 360, (0.08, 0.07, 0.07), FAKE_FRAME_THICKNESS, cv2.LINE_AA)


def synthesize_fake_blob_glare(
    clean_eye_crop: np.ndarray, lens_label_mask: np.ndarray, random_generator: np.random.Generator
) -> dict[str, np.ndarray]:
    """FAKE synthesizer: paint 1-3 colored blobs per lens and return the contract data_dict.

    Args:
        clean_eye_crop: (H, W, 3) float32 sRGB in [0, 1].
        lens_label_mask: (H, W) uint8 crop-space lens labels (0 / 1 / 2).
        random_generator: the only source of randomness.
    Returns:
        The Synthesis data_dict minus `source_id` (the dataset adds it).
    """
    height, width = lens_label_mask.shape
    lens_mask = (lens_label_mask > 0).astype(np.float32)
    reflection = np.zeros((height, width, 3), dtype=np.float32)
    if lens_mask.any() and random_generator.random() >= FAKE_NO_GLARE_PROBABILITY:
        pixel_rows, pixel_cols = np.mgrid[0:height, 0:width].astype(np.float32)
        tint = np.array(FAKE_GLARE_TINTS_RGB[random_generator.integers(len(FAKE_GLARE_TINTS_RGB))], dtype=np.float32)
        for lens_label in (LENS_LABEL_IMAGE_LEFT, LENS_LABEL_IMAGE_RIGHT):
            lens_rows, lens_cols = np.nonzero(lens_label_mask == lens_label)
            if lens_rows.size == 0:
                continue
            for _ in range(random_generator.integers(1, 4)):
                pick = random_generator.integers(lens_rows.size)
                radius = random_generator.uniform(4.0, 30.0)
                strength = random_generator.uniform(0.15, 1.5)
                squared_distance = (pixel_rows - lens_rows[pick]) ** 2 + (pixel_cols - lens_cols[pick]) ** 2
                if random_generator.random() < 0.5:
                    blob = np.exp(-squared_distance / (2.0 * radius**2))
                else:
                    blob = np.clip((radius - np.sqrt(squared_distance)) / 2.0, 0.0, 1.0)
                reflection += strength * blob[..., None] * tint
        reflection *= lens_mask[..., None]

    glare_eye_crop = np.clip(clean_eye_crop + reflection, 0.0, 1.0).astype(np.float32)
    per_pixel_change = np.abs(glare_eye_crop - clean_eye_crop).max(axis=2)
    glare_mask = np.clip(per_pixel_change / GLARE_MASK_FULL_CHANGE, 0.0, 1.0)
    lost_detail_mask = (glare_eye_crop.min(axis=2) >= LOST_DETAIL_SATURATION_LEVEL) & (
        per_pixel_change >= LOST_DETAIL_MIN_CHANGE
    )
    return {
        "clean_eye_crop": clean_eye_crop.astype(np.float32),
        "glare_eye_crop": glare_eye_crop,
        "glare_mask": glare_mask[..., None].astype(np.float32),
        "lost_detail_mask": lost_detail_mask[..., None].astype(np.float32),
        "lens_mask": lens_mask[..., None],
    }


@GLARE_SYNTHESIZER_REGISTRY.register
def build_fake_blob_glare_synthesizer() -> GlareSynthesizer:
    """Factory for config use (`GLARE_SYNTHESIZER.NAME: build_fake_blob_glare_synthesizer`)."""
    return synthesize_fake_blob_glare
