"""Write the per-image credits file for every source photo in the manifest.

Run: `python -m training_sources.write_attribution_file [--output PATH]`

CC BY 2.0 requires crediting the author, linking the work, and naming the license; public-domain
photos need no credit but get one anyway. Output is Markdown grouped by license, one line per
photo, sorted by source id, so it can be published next to released weights.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from training_sources.acquisition.ffhq_license_filter import SOURCE_LICENSE_TO_CANONICAL_URL, SourceLicense
from training_sources.load_source_manifest import SourceManifestRow, load_source_manifest
from training_sources.training_sources_paths import ATTRIBUTION_MARKDOWN_PATH, SOURCE_MANIFEST_PATH

logger = logging.getLogger(__name__)

SOURCE_LICENSE_DISPLAY_NAMES: dict[SourceLicense, str] = {
    SourceLicense.CC_BY_2_0: "Creative Commons Attribution 2.0 (CC BY 2.0)",
    SourceLicense.CC0_1_0: "CC0 1.0 Public Domain Dedication",
    SourceLicense.PUBLIC_DOMAIN_MARK_1_0: "Public Domain Mark 1.0",
    SourceLicense.US_GOVERNMENT_WORK: "United States Government Work (public domain in the US)",
}

ATTRIBUTION_HEADER = """# Source photo credits

The training and evaluation photos are a subset of the Flickr-Faces-HQ dataset (FFHQ, NVIDIA,
https://github.com/NVlabs/ffhq-dataset), which aligned and cropped each Flickr photo to 1024x1024.
Only photos whose Flickr license is CC BY 2.0, CC0 1.0, the Public Domain Mark 1.0, or a US
Government Work were used. Modifications: aligned and cropped by FFHQ; we further cropped the eye
region and painted synthetic lens reflections onto it for training.
"""


def escape_markdown_text(text: str) -> str:
    """Escape characters that would break a Markdown list line (brackets, asterisks, underscores)."""
    for special_character in ("\\", "[", "]", "*", "_", "`"):
        text = text.replace(special_character, "\\" + special_character)
    return text.replace("\n", " ").strip()


def format_attribution_line(manifest_row: SourceManifestRow) -> str:
    """Return one Markdown bullet crediting a single photo."""
    photo_title = escape_markdown_text(manifest_row.extra_fields.get("photo_title", "")) or "Untitled"
    author = escape_markdown_text(manifest_row.extra_fields.get("author", "")) or "unknown author"
    photo_url = manifest_row.extra_fields.get("photo_url", "")
    return f"- `{manifest_row.source_id}`: [{photo_title}]({photo_url}) by {author}"


def render_attribution_markdown(manifest_rows: list[SourceManifestRow]) -> str:
    """Return the full credits document, grouped by license."""
    rows_by_license: dict[SourceLicense, list[SourceManifestRow]] = defaultdict(list)
    for manifest_row in manifest_rows:
        rows_by_license[SourceLicense(manifest_row.license)].append(manifest_row)
    document_lines = [ATTRIBUTION_HEADER]
    for source_license in SourceLicense:
        license_rows = sorted(rows_by_license.get(source_license, []), key=lambda row: row.source_id)
        if not license_rows:
            continue
        license_heading = f"## {SOURCE_LICENSE_DISPLAY_NAMES[source_license]} ({len(license_rows)} photos)"
        document_lines += ["", license_heading, "", f"License: {SOURCE_LICENSE_TO_CANONICAL_URL[source_license]}", ""]
        document_lines += [format_attribution_line(row) for row in license_rows]
    return "\n".join(document_lines) + "\n"


def main() -> None:
    """Write the credits file for every manifest row."""
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--manifest", type=Path, default=SOURCE_MANIFEST_PATH)
    argument_parser.add_argument("--output", type=Path, default=ATTRIBUTION_MARKDOWN_PATH)
    arguments = argument_parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    manifest_rows = load_source_manifest(arguments.manifest)
    arguments.output.write_text(render_attribution_markdown(manifest_rows))
    logger.info("wrote credits for %d photos to %s", len(manifest_rows), arguments.output)


if __name__ == "__main__":
    main()
