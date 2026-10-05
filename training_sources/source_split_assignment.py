"""Deterministic train/val/test assignment by hashing a grouping key.

The key is the Flickr account that uploaded the photo (the user segment of the photo URL), not the
image id: FFHQ has many photos per account (event photographers, families), and the same person
often appears across one account's photos. Hashing the account keeps every face of that person in
one split. The hash is SHA-256, so the assignment is stable across machines and Python versions.
"""

import hashlib
from enum import Enum
from urllib.parse import urlparse


class SourceSplit(str, Enum):
    """Manifest `split` values."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    REAL_GLARE_EVAL = "real_glare_eval"


# Cumulative upper bounds on the hash fraction: [0, 0.90) train, [0.90, 0.95) val, [0.95, 1) test.
TRAIN_FRACTION_UPPER_BOUND = 0.90
VAL_FRACTION_UPPER_BOUND = 0.95
SPLIT_HASH_SALT = "glare-off-source-split-v1"


def flickr_account_from_photo_url(photo_url: str) -> str:
    """Return the account segment of a Flickr photo URL (`/photos/<account>/<photo id>/`).

    Falls back to the whole URL when the path does not look like a Flickr photo page, so an
    odd URL still hashes deterministically (just without account grouping).
    """
    path_segments = [segment for segment in urlparse(photo_url).path.split("/") if segment]
    if len(path_segments) >= 2 and path_segments[0] == "photos":
        return path_segments[1].lower()
    return photo_url


def hash_fraction_of_grouping_key(grouping_key: str) -> float:
    """Map a string to a stable pseudo-random number in [0, 1)."""
    digest = hashlib.sha256(f"{SPLIT_HASH_SALT}:{grouping_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split_by_grouping_key(grouping_key: str) -> SourceSplit:
    """Return the train/val/test split for a grouping key (about 90/5/5 over many keys)."""
    hash_fraction = hash_fraction_of_grouping_key(grouping_key)
    if hash_fraction < TRAIN_FRACTION_UPPER_BOUND:
        return SourceSplit.TRAIN
    if hash_fraction < VAL_FRACTION_UPPER_BOUND:
        return SourceSplit.VAL
    return SourceSplit.TEST
