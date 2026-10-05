"""Generate the Python-reference fixtures the app's JS parity tests read.

Committed (synthetic numbers only, under tests/app/fixtures/):
- eye_crop_affine_cases.json: eye-center pairs -> `compute_photo_to_eye_crop_affine` at 1x and
  2x crop size, plus `eye_crop_scale`. JS must match to 1e-6.
- area_downscale_cases.json: a synthetic image -> cv2.resize(INTER_AREA), the detector's downscale.
- eye_crop_warp_cases.json: a synthetic image -> `extract_eye_crop` and
  `warp_eye_crop_layer_back_to_photo` at a small crop size (same code path as 512x256).

Private (derived from the test photos, written to vega, never committed):
- <scratch>/yunet_reference/<photo>.json + .rgba: the photo's pixels as cv2 decodes them and
  what OpenCV's FaceDetectorYN returns on them, so the JS decode can be checked on identical input.

Run from the repo root:
    /Volumes/vega/datasets/glare-off/venv/bin/python -m tests.app.generate_app_fixtures
"""

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from eye_crop.eye_crop_geometry import (
    EYE_CROP_HEIGHT,
    EYE_CROP_WIDTH,
    compute_photo_to_eye_crop_affine,
    extract_eye_crop,
    eye_crop_scale,
    warp_eye_crop_layer_back_to_photo,
)
from eye_crop.yunet_eye_detector import (
    YUNET_MAX_DETECTION_SIDE,
    YUNET_NMS_THRESHOLD,
    YUNET_ONNX_PATH,
    YUNET_SCORE_THRESHOLD,
    YUNET_TOP_K,
    detect_face_eyes_in_photo,
)

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures"
TEST_PHOTO_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/reference-site-examples")
PRIVATE_FIXTURE_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/scratch/app-fixtures")
AFFINE_RANDOM_CASE_COUNT = 40
SMALL_CROP_WIDTH = 64
SMALL_CROP_HEIGHT = 32


def encode_array_base64(array: np.ndarray) -> str:
    """Raw little-endian bytes of a C-contiguous array, base64."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def build_affine_cases() -> list[dict]:
    """Hand-picked edge cases plus seeded random eye pairs."""
    rng = np.random.default_rng(7)
    eye_pairs = [
        ([100.0, 200.0], [300.0, 200.0]),  # level
        ([100.0, 200.0], [300.0, 260.0]),  # tilted down to the right
        ([100.0, 260.0], [300.0, 200.0]),  # tilted up to the right
        ([500.0, 500.0], [500.5, 900.0]),  # nearly vertical
        ([10.25, 10.75], [11.5, 11.0]),  # tiny eye distance (large upscale)
        ([1200.0, 3000.0], [2100.0, 2950.0]),  # big phone photo
    ]
    for _ in range(AFFINE_RANDOM_CASE_COUNT):
        left_eye = rng.uniform(0, 4000, 2)
        offset = rng.normal(0, 1, 2) * rng.uniform(5, 900)
        eye_pairs.append((left_eye.tolist(), (left_eye + np.abs(offset) * [1, np.sign(offset[1])]).tolist()))
    cases = []
    for left_eye, right_eye in eye_pairs:
        affine_1x = compute_photo_to_eye_crop_affine(np.array(left_eye), np.array(right_eye))
        affine_2x = compute_photo_to_eye_crop_affine(np.array(left_eye), np.array(right_eye), EYE_CROP_WIDTH * 2, EYE_CROP_HEIGHT * 2)
        cases.append(
            {
                "image_left_eye_xy": list(left_eye),
                "image_right_eye_xy": list(right_eye),
                "affine_1x": affine_1x.tolist(),
                "affine_2x": affine_2x.tolist(),
                "eye_crop_scale_1x": eye_crop_scale(affine_1x),
            }
        )
    return cases


def build_synthetic_photo(width: int, height: int, seed: int) -> np.ndarray:
    """A smooth-plus-texture uint8 BGR image (smooth enough that bilinear rounding is meaningful)."""
    rng = np.random.default_rng(seed)
    y_coordinates, x_coordinates = np.mgrid[0:height, 0:width].astype(np.float64)
    channels = []
    for channel_index in range(3):
        smooth = 127.5 + 100 * np.sin(x_coordinates * 0.11 + channel_index) * np.cos(y_coordinates * 0.07 - channel_index)
        channels.append(smooth + rng.normal(0, 12, (height, width)))
    return np.clip(np.stack(channels, axis=-1), 0, 255).round().astype(np.uint8)


def build_area_downscale_cases() -> list[dict]:
    """cv2.resize INTER_AREA at fractional and integer factors, called exactly as the detector calls it."""
    cases = []
    for width, height, factor in [(150, 110, 0.37), (96, 70, 0.5), (161, 53, 0.8125)]:
        photo = build_synthetic_photo(width, height, seed=width)
        resized = cv2.resize(photo, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cases.append(
            {
                "width": width,
                "height": height,
                "factor": factor,
                "photo_bgr_base64": encode_array_base64(photo),
                "resized_width": resized.shape[1],
                "resized_height": resized.shape[0],
                "resized_bgr_base64": encode_array_base64(resized),
            }
        )
    return cases


def build_eye_crop_warp_cases() -> list[dict]:
    """Photo -> crop (uint8, reflect border) and crop layer -> photo (float, zero border)."""
    photo_width, photo_height = 96, 72
    photo = build_synthetic_photo(photo_width, photo_height, seed=3)
    rng = np.random.default_rng(11)
    cases = []
    for left_eye, right_eye in [([40.0, 45.0], [80.0, 45.0]), ([30.5, 50.25], [70.0, 38.0]), ([2.0, 3.0], [30.0, 8.0])]:
        affine = compute_photo_to_eye_crop_affine(np.array(left_eye), np.array(right_eye), SMALL_CROP_WIDTH, SMALL_CROP_HEIGHT)
        crop = extract_eye_crop(photo, affine, SMALL_CROP_WIDTH, SMALL_CROP_HEIGHT)
        crop_layer = rng.normal(0, 1, (SMALL_CROP_HEIGHT, SMALL_CROP_WIDTH)).astype(np.float32)
        warped_layer = warp_eye_crop_layer_back_to_photo(crop_layer, affine, photo_width, photo_height)
        cases.append(
            {
                "image_left_eye_xy": left_eye,
                "image_right_eye_xy": right_eye,
                "affine": affine.tolist(),
                "crop_bgr_base64": encode_array_base64(crop),
                "crop_layer_float32_base64": encode_array_base64(crop_layer),
                "warped_layer_float32_base64": encode_array_base64(warped_layer),
            }
        )
    return [
        {
            "photo_width": photo_width,
            "photo_height": photo_height,
            "crop_width": SMALL_CROP_WIDTH,
            "crop_height": SMALL_CROP_HEIGHT,
            "photo_bgr_base64": encode_array_base64(photo),
            "cases": cases,
        }
    ]


def write_private_yunet_reference() -> None:
    """Per test photo: cv2-decoded RGBA pixels, raw FaceDetectorYN rows, and the eye-detector output."""
    output_directory = PRIVATE_FIXTURE_DIRECTORY / "yunet_reference"
    output_directory.mkdir(parents=True, exist_ok=True)
    for photo_path in sorted(TEST_PHOTO_DIRECTORY.glob("*-before.webp")):
        photo_bgr = cv2.imread(str(photo_path))
        height, width = photo_bgr.shape[:2]
        if max(height, width) > YUNET_MAX_DETECTION_SIDE:
            raise ValueError(f"{photo_path.name} needs downscaling; this fixture assumes native-size detection")
        detector = cv2.FaceDetectorYN.create(str(YUNET_ONNX_PATH), "", (width, height), YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K)
        _, raw_rows = detector.detect(photo_bgr)
        rgba = np.dstack([photo_bgr[:, :, ::-1], np.full((height, width), 255, np.uint8)])
        (output_directory / f"{photo_path.stem}.rgba").write_bytes(np.ascontiguousarray(rgba).tobytes())
        faces = detect_face_eyes_in_photo(photo_bgr)
        reference = {
            "photo_name": photo_path.name,
            "width": width,
            "height": height,
            "opencv_face_rows": [] if raw_rows is None else raw_rows.tolist(),
            "faces": [
                {
                    "image_left_eye_xy": face.image_left_eye_xy.tolist(),
                    "image_right_eye_xy": face.image_right_eye_xy.tolist(),
                    "detection_score": face.detection_score,
                }
                for face in faces
            ],
        }
        (output_directory / f"{photo_path.stem}.json").write_text(json.dumps(reference, indent=1))
        print(f"private yunet reference: {photo_path.stem} ({len(faces)} faces)")


def main() -> None:
    """Write every fixture."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIRECTORY / "eye_crop_affine_cases.json").write_text(json.dumps(build_affine_cases(), indent=1))
    (FIXTURE_DIRECTORY / "area_downscale_cases.json").write_text(json.dumps(build_area_downscale_cases()))
    (FIXTURE_DIRECTORY / "eye_crop_warp_cases.json").write_text(json.dumps(build_eye_crop_warp_cases()))
    print(f"wrote committed fixtures to {FIXTURE_DIRECTORY}")
    write_private_yunet_reference()


if __name__ == "__main__":
    main()
