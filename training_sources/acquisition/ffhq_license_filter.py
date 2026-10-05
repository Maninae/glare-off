"""Per-image license filter for FFHQ: keep only photos whose Flickr license allows our use.

FFHQ's `ffhq-dataset-v2.json` records each photo's Flickr license. We keep:

- CC BY 2.0 (attribution required, which the ATTRIBUTION file provides),
- CC0 1.0 and the Public Domain Mark 1.0,
- US Government Works.

Everything else (CC BY-NC 2.0, and any string we do not recognize) is dropped. The license is
matched on its canonical URL, which is less ambiguous than the display name.
"""

from enum import Enum


class SourceLicense(str, Enum):
    """The licenses a kept source photo may carry; the value is the manifest's `license` string."""

    CC_BY_2_0 = "CC-BY-2.0"
    CC0_1_0 = "CC0-1.0"
    PUBLIC_DOMAIN_MARK_1_0 = "PDM-1.0"
    US_GOVERNMENT_WORK = "US-Government-Work"


FFHQ_LICENSE_URL_TO_SOURCE_LICENSE: dict[str, SourceLicense] = {
    "https://creativecommons.org/licenses/by/2.0/": SourceLicense.CC_BY_2_0,
    "https://creativecommons.org/publicdomain/zero/1.0/": SourceLicense.CC0_1_0,
    "https://creativecommons.org/publicdomain/mark/1.0/": SourceLicense.PUBLIC_DOMAIN_MARK_1_0,
    "http://www.usa.gov/copyright.shtml": SourceLicense.US_GOVERNMENT_WORK,
}

SOURCE_LICENSE_TO_CANONICAL_URL: dict[SourceLicense, str] = {
    SourceLicense.CC_BY_2_0: "https://creativecommons.org/licenses/by/2.0/",
    SourceLicense.CC0_1_0: "https://creativecommons.org/publicdomain/zero/1.0/",
    SourceLicense.PUBLIC_DOMAIN_MARK_1_0: "https://creativecommons.org/publicdomain/mark/1.0/",
    SourceLicense.US_GOVERNMENT_WORK: "https://www.usa.gov/government-copyright",
}


def classify_ffhq_license(ffhq_license_url: str) -> SourceLicense | None:
    """Return the kept license for an FFHQ `license_url`, or None when the photo must be dropped.

    - Trailing-slash and http/https differences are normalized before matching.
    - Unknown URLs return None (fail closed: an unrecognized license is never kept).
    """
    normalized_url = ffhq_license_url.strip()
    candidate_urls = {normalized_url, normalized_url.rstrip("/") + "/", normalized_url.rstrip("/")}
    candidate_urls |= {url.replace("https://", "http://") for url in candidate_urls}
    candidate_urls |= {url.replace("http://", "https://") for url in candidate_urls}
    for candidate_url in candidate_urls:
        if candidate_url in FFHQ_LICENSE_URL_TO_SOURCE_LICENSE:
            return FFHQ_LICENSE_URL_TO_SOURCE_LICENSE[candidate_url]
    return None


def format_attribution_text(source_license: SourceLicense, author: str, photo_url: str, photo_title: str) -> str:
    """Build the one-line credit stored in the manifest's `attribution` key.

    CC BY requires the author, a link to the work, and the license; the public-domain licenses
    do not require credit, but we give it anyway.
    """
    title_part = f'"{photo_title}"' if photo_title else "Untitled photo"
    author_part = author if author else "unknown author"
    return f"{title_part} by {author_part} ({photo_url}), {source_license.value}, cropped by FFHQ (NVIDIA)"
