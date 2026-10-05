"""Atomic save and load of full training state, so an interrupted run resumes where it stopped.

A checkpoint holds the live model, EMA model, optimizer, LR scheduler, step, best val metric, and
torch/numpy/python RNG states. It is written to a temp file and renamed, so a kill mid-write
never leaves a truncated `last.pt`.

- Data order after a resume is not bit-identical (DataLoader workers reseed); everything else is.
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best_ema.pt"


def save_checkpoint_atomically(checkpoint_state: dict[str, Any], checkpoint_path: Path) -> None:
    """Write `checkpoint_state` to `checkpoint_path` via a temp file + rename."""
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(checkpoint_state, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def capture_rng_states() -> dict[str, Any]:
    """Return the python, numpy, and torch CPU RNG states."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}


def restore_rng_states(rng_states: dict[str, Any]) -> None:
    """Restore states captured by `capture_rng_states`."""
    random.setstate(rng_states["python"])
    np.random.set_state(rng_states["numpy"])
    torch.set_rng_state(rng_states["torch_cpu"])


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint onto the CPU (callers move tensors to their device via load_state_dict).

    `weights_only=False` because the state includes python/numpy RNG objects; only load
    checkpoints this project wrote.
    """
    logger.info("loading checkpoint %s", checkpoint_path)
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_model_weights_for_inference(checkpoint_path: Path, prefer_ema: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return (model state_dict, model config section) from a checkpoint, EMA weights by default."""
    checkpoint_state = load_checkpoint(checkpoint_path)
    weights_key = "ema_model" if prefer_ema and "ema_model" in checkpoint_state else "model"
    logger.info("using %s weights from step %d", weights_key, checkpoint_state["step"])
    return checkpoint_state[weights_key], checkpoint_state["model_config"]
