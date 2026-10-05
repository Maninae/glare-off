"""Load the d2go-style OmegaConf training config: `_BASE_` inheritance, config groups, dotlist overrides.

Resolution order (later wins):
1. Group files named by `CONFIG_GROUPS` (e.g. `MODEL: base` -> `configs/glare_model/model/base.yaml`).
2. The config file, after recursively merging its `_BASE_` chain.
3. CLI dotlist overrides (`SOLVER.BASE_LR=1e-3`).

`CONFIG_GROUPS` is read AFTER applying the file and the dotlist, so `CONFIG_GROUPS.MODEL=small` on
the command line swaps the model group. The loaded config is read-only and struct-locked, so a
typo in an override fails loudly instead of creating a new key.
"""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

BASE_KEY = "_BASE_"
CONFIG_GROUPS_KEY = "CONFIG_GROUPS"
CONFIG_GROUPS_DIRECTORY = Path("configs/glare_model")
MAX_DATALOADER_WORKERS = 4


def load_config_file_with_bases(config_path: Path) -> DictConfig:
    """Load `config_path`, merging its `_BASE_` file (relative to it) underneath, recursively."""
    file_config = OmegaConf.load(config_path)
    base_name = file_config.pop(BASE_KEY, None)
    if base_name is None:
        return file_config
    base_config = load_config_file_with_bases((config_path.parent / base_name).resolve())
    return OmegaConf.merge(base_config, file_config)


def load_training_config(config_path: Path, dotlist_overrides: list[str], groups_directory: Path = CONFIG_GROUPS_DIRECTORY) -> DictConfig:
    """Return the fully composed, validated, read-only training config."""
    file_config = load_config_file_with_bases(Path(config_path))
    override_config = OmegaConf.from_dotlist(dotlist_overrides)
    selected_groups = OmegaConf.merge(file_config, override_config).get(CONFIG_GROUPS_KEY, {})

    group_configs = []
    for group_name, choice in selected_groups.items():
        group_path = groups_directory / str(group_name).lower() / f"{choice}.yaml"
        if not group_path.exists():
            raise FileNotFoundError(f"config group {group_name}={choice} not found at {group_path}")
        group_configs.append(OmegaConf.load(group_path))

    composed_without_overrides = OmegaConf.merge(*group_configs, file_config)
    OmegaConf.set_struct(composed_without_overrides, True)
    training_config = OmegaConf.merge(composed_without_overrides, override_config)
    validate_training_config(training_config)
    OmegaConf.set_readonly(training_config, True)
    return training_config


def validate_training_config(training_config: DictConfig) -> None:
    """Fail loudly on settings that would break the run or the shared machine."""
    if training_config.DATALOADER.NUM_WORKERS > MAX_DATALOADER_WORKERS:
        raise ValueError(f"DATALOADER.NUM_WORKERS must be <= {MAX_DATALOADER_WORKERS} on this machine")
    for size_key in ("CROP_HEIGHT", "CROP_WIDTH"):
        if training_config.DATA[size_key] % 16 != 0:
            raise ValueError(f"DATA.{size_key} must be a multiple of 16")
    if training_config.SOLVER.WARMUP_ITERS >= training_config.SOLVER.MAX_ITER:
        raise ValueError("SOLVER.WARMUP_ITERS must be smaller than SOLVER.MAX_ITER")
