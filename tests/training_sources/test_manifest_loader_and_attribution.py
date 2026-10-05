"""Tests for the typed manifest loader, the manifest split policy, and the attribution renderer."""

import json
from pathlib import Path

import pytest

from training_sources.build_source_manifest import choose_manifest_split, load_review_rejected_source_ids
from training_sources.labeling.source_glare_score import REAL_GLARE_EVAL_SCORE_THRESHOLD, SOURCE_GLARE_SCORE_THRESHOLD
from training_sources.load_source_manifest import load_source_manifest
from training_sources.source_split_assignment import SourceSplit
from training_sources.write_attribution_file import render_attribution_markdown


def make_manifest_row(source_id: str, split: str, license_name: str = "CC-BY-2.0") -> dict:
    """Return a minimal contract-complete manifest row plus the extra keys the attribution file uses."""
    return {
        "source_id": source_id,
        "split": split,
        "photo_path": f"/photos/{source_id}.png",
        "license": license_name,
        "attribution": "credit",
        "image_left_eye_xy": [390.0, 490.0],
        "image_right_eye_xy": [630.0, 490.0],
        "lens_mask_path": f"/masks/{source_id}.png",
        "source_glare_score": 0.0,
        "has_glasses": True,
        "photo_url": f"https://www.flickr.com/photos/someone/{source_id}/",
        "author": "Some [Author]",
        "photo_title": "Title_with*marks",
    }


def write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows as a JSONL manifest and return its path."""
    manifest_path = tmp_path / "source_manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return manifest_path


def test_loader_types_fields_and_filters_by_split(tmp_path: Path) -> None:
    """Rows become typed dataclasses; split filtering keeps only the requested splits."""
    manifest_path = write_manifest(tmp_path, [make_manifest_row("a", "train"), make_manifest_row("b", "real_glare_eval")])
    all_rows = load_source_manifest(manifest_path)
    assert [row.source_id for row in all_rows] == ["a", "b"]
    assert all_rows[0].image_left_eye_xy.shape == (2,)
    assert all_rows[0].photo_path == Path("/photos/a.png")
    assert all_rows[0].extra_fields["author"] == "Some [Author]"
    assert [row.source_id for row in load_source_manifest(manifest_path, splits={"real_glare_eval"})] == ["b"]


def test_loader_fails_loud_on_missing_contract_key(tmp_path: Path) -> None:
    """A row without a contract key raises instead of silently loading."""
    broken_row = make_manifest_row("a", "train")
    del broken_row["lens_mask_path"]
    with pytest.raises(KeyError):
        load_source_manifest(write_manifest(tmp_path, [broken_row]))


def test_split_policy_uses_both_thresholds() -> None:
    """Clean -> hashed split, ambiguous -> dropped, clearly glared -> real-glare eval, negatives never eval."""
    glasses_candidate = {"has_glasses": True, "photo_url": "https://www.flickr.com/photos/x/1/"}
    negative_candidate = {"has_glasses": False, "photo_url": "https://www.flickr.com/photos/x/1/"}
    assert choose_manifest_split(glasses_candidate, SOURCE_GLARE_SCORE_THRESHOLD) in (SourceSplit.TRAIN, SourceSplit.VAL, SourceSplit.TEST)
    assert choose_manifest_split(glasses_candidate, (SOURCE_GLARE_SCORE_THRESHOLD + REAL_GLARE_EVAL_SCORE_THRESHOLD) / 2) is None
    assert choose_manifest_split(glasses_candidate, REAL_GLARE_EVAL_SCORE_THRESHOLD + 0.01) == SourceSplit.REAL_GLARE_EVAL
    assert choose_manifest_split(negative_candidate, 1.0) != SourceSplit.REAL_GLARE_EVAL


def test_review_rejection_file_ignores_comments(tmp_path: Path) -> None:
    """Ids are read one per line; `#` comments and blank lines are ignored."""
    rejections_path = tmp_path / "rejections.txt"
    rejections_path.write_text("# header\nffhq_00001  # reason\n\nffhq_00002\n")
    assert load_review_rejected_source_ids(rejections_path) == {"ffhq_00001", "ffhq_00002"}
    assert load_review_rejected_source_ids(tmp_path / "missing.txt") == set()


def test_attribution_groups_by_license_and_escapes_markdown(tmp_path: Path) -> None:
    """Each photo gets a credit line under its license heading, with Markdown specials escaped."""
    manifest_path = write_manifest(tmp_path, [make_manifest_row("a", "train"), make_manifest_row("b", "test", "CC0-1.0")])
    attribution_markdown = render_attribution_markdown(load_source_manifest(manifest_path))
    assert "## Creative Commons Attribution 2.0 (CC BY 2.0) (1 photos)" in attribution_markdown
    assert "## CC0 1.0 Public Domain Dedication (1 photos)" in attribution_markdown
    assert "by Some \\[Author\\]" in attribution_markdown
    assert "Title\\_with\\*marks" in attribution_markdown
