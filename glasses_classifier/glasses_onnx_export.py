"""Export GlassesClassifierNet to the app's ONNX contract, verify it, and report.

Contract (glasses_classifier/CLAUDE.md, "Model I/O"): opset 17, input `eye_crop` float32
[1, 3, 256, 512], output `glasses_probability` float32 [1, 1] (sigmoid inside the graph), only
onnxruntime-web WebGPU ops, fp16 weights when parity holds.

Reuses the glare model's export machinery (`glare_model/export/`): dynamo at opset 18, the
hand-written downconvert to 17, debug-metadata strip, WebGPU op check, fp16-weights-with-Cast.
The shapes are static: the app always feeds the fixed 256x512 crop.
"""

import copy
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from glare_model.export.onnx_export import convert_initializers_to_fp16_with_cast, save_onnx_model, strip_exporter_debug_metadata
from glare_model.export.onnx_verification import create_single_thread_cpu_session
from glare_model.export.opset_downconversion import SOURCE_OPSET, downconvert_opset18_model_to_opset17
from glare_model.export.webgpu_operator_inventory import check_graph_uses_only_webgpu_operators
from glasses_classifier.glasses_classifier_net import GlassesClassifierNet, count_trainable_parameters

logger = logging.getLogger(__name__)

INPUT_NAME = "eye_crop"
OUTPUT_NAME = "glasses_probability"
CROP_HEIGHT = 256
CROP_WIDTH = 512
PARITY_SEED = 1234
PARITY_SAMPLE_COUNT = 8
FP32_PARITY_TOLERANCE = 1e-4
# A probability shift of 0.002 cannot move a face across any sane threshold except by a hair.
FP16_PARITY_TOLERANCE = 2e-3
LATENCY_WARMUP_RUNS = 3
LATENCY_TIMED_RUNS = 20


def export_glasses_classifier_to_onnx_model(model: GlassesClassifierNet) -> onnx.ModelProto:
    """Export a CPU eval copy of `model` to a checked opset-17 ModelProto (caller's model untouched).

    Raises:
        UnsupportedWebGPUOperatorError: if the graph contains an op outside the WebGPU table.
    """
    export_model = copy.deepcopy(model).to("cpu").eval()
    sample_eye_crop = torch.rand(1, 3, CROP_HEIGHT, CROP_WIDTH)
    with torch.no_grad():
        export_program = torch.onnx.export(
            export_model,
            (sample_eye_crop,),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=SOURCE_OPSET,
            dynamo=True,
            verbose=False,
        )
    opset17_model = downconvert_opset18_model_to_opset17(export_program.model_proto)
    strip_exporter_debug_metadata(opset17_model)
    onnx.checker.check_model(opset17_model, full_check=True)
    check_graph_uses_only_webgpu_operators(opset17_model)
    return opset17_model


def run_onnx_and_torch_on_fixed_crops(model: GlassesClassifierNet, onnx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (torch probabilities, onnx probabilities), each [PARITY_SAMPLE_COUNT], on fixed random-but-structured crops.

    Crops are smoothed noise rather than white noise so the probabilities spread away from one value.
    """
    cpu_model = copy.deepcopy(model).to("cpu").eval()
    session = create_single_thread_cpu_session(onnx_path)
    random_generator = np.random.default_rng(PARITY_SEED)
    torch_probabilities, onnx_probabilities = [], []
    for _ in range(PARITY_SAMPLE_COUNT):
        coarse_noise = random_generator.random((1, 3, CROP_HEIGHT // 32, CROP_WIDTH // 32), dtype=np.float32)
        eye_crop = np.repeat(np.repeat(coarse_noise, 32, axis=2), 32, axis=3)
        eye_crop = np.clip(eye_crop + 0.1 * random_generator.standard_normal(eye_crop.shape, dtype=np.float32), 0, 1)
        with torch.no_grad():
            torch_probabilities.append(float(cpu_model(torch.from_numpy(eye_crop))[0, 0]))
        (onnx_output,) = session.run([OUTPUT_NAME], {INPUT_NAME: eye_crop})
        if onnx_output.shape != (1, 1):
            raise AssertionError(f"ONNX output shape {onnx_output.shape} breaks the [1, 1] contract")
        onnx_probabilities.append(float(onnx_output[0, 0]))
    return np.array(torch_probabilities), np.array(onnx_probabilities)


def measure_single_thread_cpu_latency_ms(onnx_path: Path) -> float:
    """Median ms of one 256x512 inference on onnxruntime's CPU EP with one thread (rough phone-WASM proxy)."""
    session = create_single_thread_cpu_session(onnx_path)
    eye_crop = np.random.default_rng(PARITY_SEED).random((1, 3, CROP_HEIGHT, CROP_WIDTH), dtype=np.float32)
    for _ in range(LATENCY_WARMUP_RUNS):
        session.run([OUTPUT_NAME], {INPUT_NAME: eye_crop})
    durations_ms = []
    for _ in range(LATENCY_TIMED_RUNS):
        start_time = time.perf_counter()
        session.run([OUTPUT_NAME], {INPUT_NAME: eye_crop})
        durations_ms.append((time.perf_counter() - start_time) * 1000.0)
    return float(np.median(durations_ms))


def export_and_verify_glasses_classifier(model: GlassesClassifierNet, output_path: Path, prefer_fp16_weights: bool = True) -> dict[str, Any]:
    """Export to `output_path` (fp16 weights if parity holds, else fp32), verify, and return a JSON-friendly report.

    - The fp32 graph is always built and parity-checked first; when `prefer_fp16_weights` and the
      fp16-weights variant stays within FP16_PARITY_TOLERANCE, that variant is what lands at
      `output_path` (the app ships one file).
    Raises:
        AssertionError: fp32 parity exceeds FP32_PARITY_TOLERANCE.
    """
    fp32_model = export_glasses_classifier_to_onnx_model(model)
    with tempfile.TemporaryDirectory() as temporary_directory:
        fp32_path = Path(temporary_directory) / "glasses_classifier_fp32.onnx"
        fp32_size_bytes = save_onnx_model(fp32_model, fp32_path)
        torch_probabilities, fp32_probabilities = run_onnx_and_torch_on_fixed_crops(model, fp32_path)
        fp32_max_diff = float(np.abs(torch_probabilities - fp32_probabilities).max())
        if fp32_max_diff > FP32_PARITY_TOLERANCE:
            raise AssertionError(f"fp32 ONNX vs PyTorch max abs diff {fp32_max_diff:.2e} exceeds {FP32_PARITY_TOLERANCE}")

        chosen_path, fp16_report = fp32_path, None
        if prefer_fp16_weights:
            fp16_path = Path(temporary_directory) / "glasses_classifier_fp16.onnx"
            fp16_size_bytes = save_onnx_model(convert_initializers_to_fp16_with_cast(fp32_model), fp16_path)
            _, fp16_probabilities = run_onnx_and_torch_on_fixed_crops(model, fp16_path)
            fp16_max_diff = float(np.abs(torch_probabilities - fp16_probabilities).max())
            fp16_keeps_parity = fp16_max_diff <= FP16_PARITY_TOLERANCE
            fp16_report = {"file_size_mb": fp16_size_bytes / 1e6, "max_abs_diff": fp16_max_diff, "keeps_parity": fp16_keeps_parity}
            if fp16_keeps_parity:
                chosen_path = fp16_path
            else:
                logger.warning("fp16 weights diff %.2e > %.2e: shipping fp32 weights", fp16_max_diff, FP16_PARITY_TOLERANCE)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen_path, output_path)

    shipped_model = onnx.load(str(output_path))
    return {
        "onnx_path": str(output_path),
        "weights": "fp16" if chosen_path.name.endswith("fp16.onnx") else "fp32",
        "parameter_count": count_trainable_parameters(model),
        "file_size_mb": output_path.stat().st_size / 1e6,
        "fp32_file_size_mb": fp32_size_bytes / 1e6,
        "opset": next(entry.version for entry in shipped_model.opset_import if entry.domain == ""),
        "operator_counts": check_graph_uses_only_webgpu_operators(shipped_model),
        "webgpu_operator_check": "pass",
        "parity_fp32_max_abs_diff": fp32_max_diff,
        "parity_fp16": fp16_report,
        "sample_probabilities_torch": [round(value, 4) for value in torch_probabilities.tolist()],
        "cpu_single_thread_latency_ms": measure_single_thread_cpu_latency_ms(output_path),
    }
