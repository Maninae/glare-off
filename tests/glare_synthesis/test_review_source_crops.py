"""The manifest path of the review loader: rows read with training_sources' loader, cropped, masks warped."""

import json

import cv2
import numpy as np

from glare_synthesis.render_lens_glare import render_lens_glare_sample
from glare_synthesis.review_source_crops import load_crops_from_source_manifest

PHOTO_SIZE = 600


def write_tiny_source_manifest(directory) -> tuple:
    """One synthetic face photo with two lens blobs, its label mask, and a one-row manifest."""
    photo_bgr = np.full((PHOTO_SIZE, PHOTO_SIZE, 3), (110, 140, 190), np.uint8)
    lens_label_map = np.zeros((PHOTO_SIZE, PHOTO_SIZE), np.uint8)
    eye_centers = {1: (230, 300), 2: (370, 300)}
    for lens_label, eye_center in eye_centers.items():
        cv2.circle(photo_bgr, eye_center, 10, (40, 30, 30), -1)
        cv2.ellipse(lens_label_map, eye_center, (55, 40), 0, 0, 360, lens_label, -1)
    photo_path, mask_path = directory / "face.png", directory / "face__lens_mask.png"
    cv2.imwrite(str(photo_path), photo_bgr)
    cv2.imwrite(str(mask_path), lens_label_map)
    manifest_row = {
        "source_id": "test_face", "split": "train", "photo_path": str(photo_path), "license": "CC0", "attribution": "synthetic",
        "image_left_eye_xy": list(eye_centers[1]), "image_right_eye_xy": list(eye_centers[2]),
        "lens_mask_path": str(mask_path), "source_glare_score": 0.0, "has_glasses": True,
    }
    manifest_path = directory / "source_manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest_row) + "\n")
    return manifest_path


def test_manifest_crops_have_both_lenses_and_render(tmp_path):
    manifest_path = write_tiny_source_manifest(tmp_path)
    review_crops = load_crops_from_source_manifest(manifest_path, maximum_crop_count=4, random_generator=np.random.default_rng(0))
    assert len(review_crops) == 1
    review_crop = review_crops[0]
    assert review_crop.clean_eye_crop.shape == (256, 512, 3) and review_crop.clean_eye_crop.dtype == np.float32
    assert set(np.unique(review_crop.lens_label_mask)) == {0, 1, 2}
    # Image-left lens label lands on the left half of the crop.
    left_lens_columns = np.nonzero(review_crop.lens_label_mask == 1)[1]
    assert left_lens_columns.mean() < 256
    data_dict = render_lens_glare_sample(review_crop.clean_eye_crop, review_crop.lens_label_mask, np.random.default_rng(1))
    assert data_dict["source_id"] == ""
    assert data_dict["lens_mask"].sum() > 0
