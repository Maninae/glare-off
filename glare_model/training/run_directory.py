"""Hash-named run directories: `<RUNS_ROOT>/run_output_<hash>/{tb,checkpoints,exported}` + config.yaml.

The hash covers the resolved config minus keys that do not change what is learned (worker count,
resume flag, device choice, logging cadence), so rerunning the same command lands in the same
directory and resumes from its last checkpoint instead of starting a twin run.
`OUTPUT.RUN_DIRECTORY` (non-empty) overrides the hashed name entirely.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

RUN_HASH_LENGTH = 10
KEYS_EXCLUDED_FROM_RUN_HASH = (
    ("DATALOADER", "NUM_WORKERS"),
    ("TRAIN", "RESUME"),
    ("TRAIN", "DEVICE"),
    ("TRAIN", "MPS_HIGH_WATERMARK_RATIO"),
    ("TRAIN", "MPS_LOW_WATERMARK_RATIO"),
    ("TRAIN", "LOG_PERIOD"),
    ("TRAIN", "IMAGE_GRID_COUNT"),
    ("OUTPUT", "RUN_DIRECTORY"),
)


@dataclass
class RunDirectory:
    """Paths of one training run."""

    root: Path
    tensorboard_directory: Path
    checkpoint_directory: Path
    exported_directory: Path
    config_path: Path


def compute_run_hash(training_config: DictConfig) -> str:
    """Return a short sha256 of the config with `KEYS_EXCLUDED_FROM_RUN_HASH` removed."""
    hashable_config = OmegaConf.to_container(training_config, resolve=True)
    for section, key in KEYS_EXCLUDED_FROM_RUN_HASH:
        hashable_config.get(section, {}).pop(key, None)
    canonical_json = json.dumps(hashable_config, sort_keys=True)
    return hashlib.sha256(canonical_json.encode()).hexdigest()[:RUN_HASH_LENGTH]


def prepare_run_directory(training_config: DictConfig) -> RunDirectory:
    """Create the run directory tree and save the resolved config into it."""
    explicit_directory = training_config.OUTPUT.RUN_DIRECTORY
    if explicit_directory:
        root = Path(explicit_directory)
    else:
        root = Path(training_config.OUTPUT.RUNS_ROOT) / f"run_output_{compute_run_hash(training_config)}"
    run_directory = RunDirectory(
        root=root,
        tensorboard_directory=root / "tb",
        checkpoint_directory=root / "checkpoints",
        exported_directory=root / "exported",
        config_path=root / "config.yaml",
    )
    for directory in (run_directory.tensorboard_directory, run_directory.checkpoint_directory, run_directory.exported_directory):
        directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(training_config, run_directory.config_path, resolve=True)
    logger.info("run directory: %s", root)
    return run_directory
