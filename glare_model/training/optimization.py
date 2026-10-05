"""Optimizer, warmup-cosine LR schedule, EMA weights, device choice, and seeding.

Small, stateless builders the trainer composes; each returns a standard torch object so the
trainer can checkpoint it with `state_dict()`.
"""

import copy
import logging
import math
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)


def apply_mps_memory_cap(high_watermark_ratio: float, low_watermark_ratio: float) -> None:
    """Cap the MPS allocator at `high_watermark_ratio` of `torch.mps.recommended_max_memory()`.

    Must run before the first MPS allocation (the allocator reads these env vars lazily at init).
    An existing env setting wins, so a launcher can tighten it further.
    - torch 2.14 rejects a high ratio below the default LOW ratio (1.4) with "invalid low watermark
      ratio", so both are set; 0.2 on this 16 GB M4 is 2.37 GiB.
    """
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(high_watermark_ratio))
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(low_watermark_ratio))


def read_peak_accelerator_memory_gib(device: torch.device) -> float:
    """Current MPS driver allocation (or CUDA max allocated) in GiB; 0 on CPU."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    return 0.0


def select_training_device(device_setting: str) -> torch.device:
    """Return the device for `TRAIN.DEVICE`: "auto" picks CUDA, then MPS, then CPU."""
    if device_setting != "auto":
        return torch.device(device_setting)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (all devices) from `GLOBAL.SEED`."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_adamw_optimizer(model: nn.Module, solver_config: DictConfig) -> torch.optim.AdamW:
    """AdamW with weight decay on conv weights only (not biases, norm affines, residual scales)."""
    decayed_parameters = [parameter for parameter in model.parameters() if parameter.ndim == 4]
    undecayed_parameters = [parameter for parameter in model.parameters() if parameter.ndim != 4]
    optimizer_config = solver_config.OPTIMIZER
    return torch.optim.AdamW(
        [
            {"params": decayed_parameters, "weight_decay": optimizer_config.WEIGHT_DECAY},
            {"params": undecayed_parameters, "weight_decay": 0.0},
        ],
        lr=solver_config.BASE_LR,
        betas=tuple(optimizer_config.BETAS),
    )


def build_warmup_cosine_scheduler(optimizer: torch.optim.Optimizer, solver_config: DictConfig) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from ~0 to BASE_LR over WARMUP_ITERS, then cosine down to MIN_LR at MAX_ITER."""
    warmup_iters = solver_config.WARMUP_ITERS
    max_iter = solver_config.MAX_ITER
    min_lr_ratio = solver_config.MIN_LR / solver_config.BASE_LR

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        progress = min((step - warmup_iters) / max(max_iter - warmup_iters, 1), 1.0)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)


class ExponentialMovingAverageModel:
    """A frozen copy of the model whose weights track `decay * ema + (1 - decay) * live`.

    Validation and export use the EMA copy: it is smoother than the live weights and usually a
    few tenths of a dB better on restoration tasks. Buffers (none in this network) are copied.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.averaged_model = copy.deepcopy(model).eval()
        for parameter in self.averaged_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Move each averaged parameter toward the live one."""
        for averaged_parameter, live_parameter in zip(self.averaged_model.parameters(), model.parameters()):
            averaged_parameter.lerp_(live_parameter.detach(), 1.0 - self.decay)
        for averaged_buffer, live_buffer in zip(self.averaged_model.buffers(), model.buffers()):
            averaged_buffer.copy_(live_buffer)
