"""Tests for the FFHQ license filter, glasses-label agreement, and the slim metadata streaming parser."""

import json

import pytest

from training_sources.acquisition.ffhq_glasses_labels import GlassesCategory, combine_glasses_labels
from training_sources.acquisition.ffhq_license_filter import SourceLicense, classify_ffhq_license, format_attribution_text
from training_sources.acquisition.ffhq_metadata_slim_cache import dlib_eye_centers_from_ffhq_landmarks, iterate_top_level_json_object_items


@pytest.mark.parametrize(
    ("license_url", "expected_license"),
    [
        ("https://creativecommons.org/licenses/by/2.0/", SourceLicense.CC_BY_2_0),
        ("https://creativecommons.org/publicdomain/zero/1.0/", SourceLicense.CC0_1_0),
        ("https://creativecommons.org/publicdomain/mark/1.0/", SourceLicense.PUBLIC_DOMAIN_MARK_1_0),
        ("http://www.usa.gov/copyright.shtml", SourceLicense.US_GOVERNMENT_WORK),
        # Scheme and trailing-slash variants still match.
        ("http://creativecommons.org/licenses/by/2.0", SourceLicense.CC_BY_2_0),
    ],
)
def test_kept_licenses_are_recognized(license_url: str, expected_license: SourceLicense) -> None:
    """Every license we keep maps to its SourceLicense, regardless of URL formatting."""
    assert classify_ffhq_license(license_url) == expected_license


@pytest.mark.parametrize(
    "license_url",
    [
        "https://creativecommons.org/licenses/by-nc/2.0/",
        "https://creativecommons.org/licenses/by-nc-sa/2.0/",
        "https://creativecommons.org/licenses/by-nd/2.0/",
        "",
        "https://example.com/some-unknown-license",
    ],
)
def test_non_commercial_and_unknown_licenses_are_dropped(license_url: str) -> None:
    """NC and unrecognized licenses fail closed (None = drop the photo)."""
    assert classify_ffhq_license(license_url) is None


def test_attribution_text_names_author_link_and_license() -> None:
    """CC BY needs author, link, and license in the credit line."""
    attribution = format_attribution_text(SourceLicense.CC_BY_2_0, "Ada Example", "https://www.flickr.com/photos/ada/1/", "Portrait")
    assert "Ada Example" in attribution
    assert "https://www.flickr.com/photos/ada/1/" in attribution
    assert "CC-BY-2.0" in attribution


@pytest.mark.parametrize(
    ("dcgm_label", "aging_label", "expected_category"),
    [
        ("ReadingGlasses", "Normal", GlassesCategory.CLEAR_GLASSES),
        ("NoGlasses", "None", GlassesCategory.NO_GLASSES),
        ("ReadingGlasses", "Dark", GlassesCategory.DISAGREEING_OR_DARK),
        ("Sunglasses", "Normal", GlassesCategory.DISAGREEING_OR_DARK),
        ("NoGlasses", "Normal", GlassesCategory.DISAGREEING_OR_DARK),
        ("missing", "Normal", GlassesCategory.DISAGREEING_OR_DARK),
    ],
)
def test_glasses_category_requires_both_labelers_to_agree(dcgm_label: str, aging_label: str, expected_category: GlassesCategory) -> None:
    """Only agreement yields a positive or a negative; anything else is set aside."""
    assert combine_glasses_labels(dcgm_label, aging_label) == expected_category


def test_streaming_parser_matches_json_load() -> None:
    """The one-entry-at-a-time parser yields exactly what json.load would."""
    document = {"0": {"a": [1, 2, {"b": "x,}:"}]}, "1": {"a": []}, "10": {"nested": {"k": None}}}
    json_text = json.dumps(document, indent=1)
    assert dict(iterate_top_level_json_object_items(json_text)) == document


def test_dlib_eye_centers_are_ordered_by_image_x() -> None:
    """Points 36-41 and 42-47 average to the two eye centers, image-left first."""
    landmarks = [[0.0, 0.0]] * 68
    landmarks = [list(point) for point in landmarks]
    for point_index in range(36, 42):
        landmarks[point_index] = [600.0, 500.0]  # subject's left eye, on the image right here
    for point_index in range(42, 48):
        landmarks[point_index] = [400.0, 502.0]
    image_left_eye, image_right_eye = dlib_eye_centers_from_ffhq_landmarks(landmarks)
    assert image_left_eye == [400.0, 502.0]
    assert image_right_eye == [600.0, 500.0]
