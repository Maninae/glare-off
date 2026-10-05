"""Verify an exported ONNX graph against PyTorch and measure a rough CPU latency.

- Parity runs the same random crop through PyTorch (fp32, CPU) and onnxruntime and reports the
  max absolute difference per output, at every requested size.
- Latency uses onnxruntime's CPU EP pinned to one thread: a rough stand-in for single-threaded
  WASM on a phone (the floor the app designs for), NOT a browser measurement.
"""

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet
from glare_model.export.onnx_export import INPUT_NAME, OUTPUT_NAMES

logger = logging.getLogger(__name__)

PARITY_SEED = 1234
LATENCY_WARMUP_RUNS = 2
LATENCY_TIMED_RUNS = 5


@dataclass
class OnnxParityResult:
    """Max |torch - onnxruntime| per output at one input size."""

    height: int
    width: int
    clean_crop_max_abs_diff: float
    masks_max_abs_diff: float


def create_single_thread_cpu_session(onnx_path: Path) -> onnxruntime.InferenceSession:
    """Open `onnx_path` on the CPU EP with one intra-op and one inter-op thread."""
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(str(onnx_path), session_options, providers=["CPUExecutionProvider"])


def measure_onnx_parity(
    model: GlareRemovalNAFNet, onnx_path: Path, crop_sizes: list[tuple[int, int]]
) -> list[OnnxParityResult]:
    """Compare PyTorch and onnxruntime outputs on a fixed random crop at each (height, width)."""
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    session = create_single_thread_cpu_session(onnx_path)
    random_generator = np.random.default_rng(PARITY_SEED)
    parity_results = []
    for height, width in crop_sizes:
        glare_crop = random_generator.random((1, 3, height, width), dtype=np.float32)
        with torch.no_grad():
            torch_clean_crop, torch_masks = cpu_model(torch.from_numpy(glare_crop))
        onnx_clean_crop, onnx_masks = session.run(list(OUTPUT_NAMES), {INPUT_NAME: glare_crop})
        if onnx_clean_crop.shape != (1, 3, height, width) or onnx_masks.shape != (1, 2, height, width):
            raise AssertionError(f"ONNX output shapes {onnx_clean_crop.shape}, {onnx_masks.shape} break the contract")
        parity_results.append(
            OnnxParityResult(
                height=height,
                width=width,
                clean_crop_max_abs_diff=float(np.abs(torch_clean_crop.numpy() - onnx_clean_crop).max()),
                masks_max_abs_diff=float(np.abs(torch_masks.numpy() - onnx_masks).max()),
            )
        )
    return parity_results


def measure_single_thread_cpu_latency_ms(onnx_path: Path, height: int, width: int) -> float:
    """Return the median wall-clock ms of one inference at (height, width), one CPU thread."""
    session = create_single_thread_cpu_session(onnx_path)
    glare_crop = np.random.default_rng(PARITY_SEED).random((1, 3, height, width), dtype=np.float32)
    for _ in range(LATENCY_WARMUP_RUNS):
        session.run(list(OUTPUT_NAMES), {INPUT_NAME: glare_crop})
    run_durations_ms = []
    for _ in range(LATENCY_TIMED_RUNS):
        start_time = time.perf_counter()
        session.run(list(OUTPUT_NAMES), {INPUT_NAME: glare_crop})
        run_durations_ms.append((time.perf_counter() - start_time) * 1000.0)
    return float(np.median(run_durations_ms))
