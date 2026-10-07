"""Load the glasses-classifier OmegaConf config: one YAML file plus CLI dotlist overrides.

The loaded config is struct-locked and read-only, so a typo in an override fails loudly instead
of silently creating a new key. Validation enforces the shared-machine limits.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

DEFAULT_CONFIG_PATH = Path("configs/glasses_classifier/base.yaml")
# Shared 16 GB machine: each DataLoader worker costs RAM, and the cache read is not the bottleneck.
MAX_DATALOADER_WORKERS = 2
MAX_CACHE_BUILD_PROCESSES = 4


def load_glasses_classifier_config(config_path: Path = DEFAULT_CONFIG_PATH, dotlist_overrides: list[str] | None = None) -> DictConfig:
    """Return the composed, validated, read-only config."""
    file_config = OmegaConf.load(config_path)
    OmegaConf.set_struct(file_config, True)
    config = OmegaConf.merge(file_config, OmegaConf.from_dotlist(list(dotlist_overrides or [])))
    validate_glasses_classifier_config(config)
    OmegaConf.set_readonly(config, True)
    return config


def validate_glasses_classifier_config(config: DictConfig) -> None:
    """Fail loudly on settings that would break the run or overload the shared machine."""
    if config.DATALOADER.NUM_WORKERS > MAX_DATALOADER_WORKERS:
        raise ValueError(f"DATALOADER.NUM_WORKERS must be <= {MAX_DATALOADER_WORKERS} on this machine")
    if config.CACHE.BUILD_PROCESSES > MAX_CACHE_BUILD_PROCESSES:
        raise ValueError(f"CACHE.BUILD_PROCESSES must be <= {MAX_CACHE_BUILD_PROCESSES}")
    if config.CACHE.TRAIN_VARIANTS < 2:
        raise ValueError("CACHE.TRAIN_VARIANTS must be >= 2 (the last variant is the phone-capture one)")
    if config.SOLVER.WARMUP_EPOCHS >= config.SOLVER.EPOCHS:
        raise ValueError("SOLVER.WARMUP_EPOCHS must be smaller than SOLVER.EPOCHS")
