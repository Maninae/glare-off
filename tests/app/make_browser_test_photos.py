"""Build the end-to-end browser test photos and the Python detector's answers for them.

Everything is derived from the private reference photos and written to vega, never the repo:
    <scratch>/app-e2e/<name>.png                 lossless copy (the bit-exactness check needs PNG out)
    <scratch>/app-e2e/rotated-exif6.jpg          stored sideways with EXIF Orientation=6
    <scratch>/app-e2e/large-27mp.jpg             a 6000x4500 upscale (exercises the >1024 px
                                                 INTER_AREA detection path and full-res export)
    <scratch>/app-e2e/python_reference.json      eye centers from eye_crop.yunet_eye_detector
                                                 for every test file, in display orientation

Run from the repo root:
    /Volumes/vega/datasets/glare-off/venv/bin/python -m tests.app.make_browser_test_photos
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from eye_crop.yunet_eye_detector import detect_face_eyes_in_photo

TEST_PHOTO_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/reference-site-examples")
OUTPUT_DIRECTORY = Path("/Volumes/vega/datasets/glare-off/scratch/app-e2e")
EXIF_ORIENTATION_TAG = 0x0112
EXIF_ROTATE_90_CLOCKWISE = 6
LARGE_PHOTO_SIZE = (6000, 4500)
JPEG_QUALITY = 92


def describe_faces(photo_bgr: np.ndarray) -> list[dict]:
    """Python reference detections for one upright BGR photo."""
    return [
        {"image_left_eye_xy": face.image_left_eye_xy.tolist(), "image_right_eye_xy": face.image_right_eye_xy.tolist(), "detection_score": face.detection_score}
        for face in detect_face_eyes_in_photo(photo_bgr)
    ]


def main() -> None:
    """Write every test photo and the reference JSON."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    reference = {}
    for photo_path in sorted(TEST_PHOTO_DIRECTORY.glob("*-before.webp")):
        photo_bgr = cv2.imread(str(photo_path))
        reference[photo_path.name] = {"width": photo_bgr.shape[1], "height": photo_bgr.shape[0], "faces": describe_faces(photo_bgr)}
        png_name = photo_path.stem + ".png"
        cv2.imwrite(str(OUTPUT_DIRECTORY / png_name), photo_bgr)
        reference[png_name] = reference[photo_path.name]

    upright_bgr = cv2.imread(str(TEST_PHOTO_DIRECTORY / "heavy-reflection-glasses-before.webp"))
    # Orientation 6 means "rotate 90 degrees clockwise to display", so store it rotated counter-clockwise.
    stored_sideways_rgb = cv2.cvtColor(cv2.rotate(upright_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)
    exif = Image.Exif()
    exif[EXIF_ORIENTATION_TAG] = EXIF_ROTATE_90_CLOCKWISE
    rotated_path = OUTPUT_DIRECTORY / "rotated-exif6.jpg"
    Image.fromarray(stored_sideways_rgb).save(rotated_path, quality=JPEG_QUALITY, exif=exif.tobytes())
    displayed_bgr = cv2.imread(str(rotated_path))  # cv2.imread applies EXIF orientation by default
    reference[rotated_path.name] = {"width": displayed_bgr.shape[1], "height": displayed_bgr.shape[0], "faces": describe_faces(displayed_bgr)}

    large_path = OUTPUT_DIRECTORY / "large-27mp.jpg"
    large_bgr = cv2.resize(upright_bgr, LARGE_PHOTO_SIZE, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(large_path), large_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    large_decoded_bgr = cv2.imread(str(large_path))
    reference[large_path.name] = {"width": large_decoded_bgr.shape[1], "height": large_decoded_bgr.shape[0], "faces": describe_faces(large_decoded_bgr)}

    (OUTPUT_DIRECTORY / "python_reference.json").write_text(json.dumps(reference, indent=1))
    for name, entry in reference.items():
        print(f"{name}: {entry['width']}x{entry['height']}, {len(entry['faces'])} faces")


if __name__ == "__main__":
    main()
