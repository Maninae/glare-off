"""Save and load glasses-classifier checkpoints (weights + the config that built them)."""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from glasses_classifier.glasses_classifier_net import GlassesClassifierNet, build_glasses_classifier_from_config


def save_glasses_classifier_checkpoint(model: GlassesClassifierNet, config: DictConfig, epoch: int, val_metrics: dict, checkpoint_path: Path) -> None:
    """Atomically write {model_state, config, epoch, val_metrics} to `checkpoint_path`."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(
        {
            "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "config": OmegaConf.to_container(config, resolve=True),
            "epoch": epoch,
            "val_metrics": val_metrics,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def load_glasses_classifier_checkpoint(checkpoint_path: Path) -> tuple[GlassesClassifierNet, dict]:
    """Return (eval-mode CPU model, the checkpoint's config dict)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_glasses_classifier_from_config(checkpoint["config"]["MODEL"])
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint["config"]
