"""JSONL reading helpers shared by the stages that consume other stages' outputs."""

import json
from pathlib import Path


def load_jsonl_rows(jsonl_path: Path) -> list[dict]:
    """Read every non-empty line of a JSONL file."""
    with open(jsonl_path) as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def latest_annotation_row_per_source(annotation_rows: list[dict]) -> list[dict]:
    """Keep only the last row written for each source id (a rerun appends; the newest wins)."""
    rows_by_source_id = {row["source_id"]: row for row in annotation_rows}
    return list(rows_by_source_id.values())
