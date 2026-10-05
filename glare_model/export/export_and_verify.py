"""Export a model, verify it, and produce a report: the shared core of `export_onnx` and `make_stub_onnx`.

Steps: contract export (opset 17, dynamic H/W, WebGPU op check) -> parity vs PyTorch at 256x512
and 512x1024 -> optional fp16-weights variant, kept only if it also passes its parity bound ->
single-thread CPU latency at 256x512.
"""

import logging
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import onnx

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet, count_trainable_parameters
from glare_model.export.onnx_export import (
    convert_initializers_to_fp16_with_cast,
    export_glare_model_to_onnx_model,
    save_onnx_model,
)
from glare_model.export.onnx_verification import measure_onnx_parity, measure_single_thread_cpu_latency_ms
from glare_model.export.webgpu_operator_inventory import count_graph_operators, describe_operator_caveats

logger = logging.getLogger(__name__)

PARITY_CROP_SIZES = [(256, 512), (512, 1024)]
FP32_PARITY_TOLERANCE = 1e-3
# fp16 weights shift outputs by roughly the fp16 rounding of each weight; 1/255 is one 8-bit level.
FP16_PARITY_TOLERANCE = 1.0 / 255.0
FP16_FILE_SUFFIX = "_fp16"


def export_and_verify_glare_model(model: GlareRemovalNAFNet, output_path: Path, write_fp16_variant: bool = True) -> dict[str, Any]:
    """Export `model` to `output_path`, verify it, and return a JSON-friendly report.

    Raises:
        AssertionError: fp32 parity exceeds `FP32_PARITY_TOLERANCE` at any size.
        UnsupportedWebGPUOperatorError: via the export, on any op outside the WebGPU table.
    """
    onnx_model = export_glare_model_to_onnx_model(model)
    file_size_bytes = save_onnx_model(onnx_model, output_path)
    operator_counts = count_graph_operators(onnx_model)
    parity_results = measure_onnx_parity(model, output_path, PARITY_CROP_SIZES)
    worst_fp32_diff = max(max(result.clean_crop_max_abs_diff, result.masks_max_abs_diff) for result in parity_results)
    if worst_fp32_diff > FP32_PARITY_TOLERANCE:
        raise AssertionError(f"ONNX vs PyTorch max abs diff {worst_fp32_diff:.2e} exceeds {FP32_PARITY_TOLERANCE}")

    export_report: dict[str, Any] = {
        "onnx_path": str(output_path),
        "parameter_count": count_trainable_parameters(model),
        "file_size_mb": file_size_bytes / 1e6,
        "opset": next(entry.version for entry in onnx_model.opset_import if entry.domain == ""),
        "operator_counts": operator_counts,
        "operator_caveats": describe_operator_caveats(operator_counts),
        "webgpu_operator_check": "pass",
        "parity_fp32": [asdict(result) for result in parity_results],
        "cpu_single_thread_latency_ms_256x512": measure_single_thread_cpu_latency_ms(output_path, 256, 512),
    }

    if write_fp16_variant:
        export_report["fp16_weights_variant"] = export_fp16_variant_if_it_keeps_parity(model, onnx_model, output_path)
    return export_report


def export_fp16_variant_if_it_keeps_parity(model: GlareRemovalNAFNet, onnx_model: onnx.ModelProto, output_path: Path) -> dict[str, Any]:
    """Write `<stem>_fp16.onnx` next to `output_path` only if its parity is within `FP16_PARITY_TOLERANCE`.

    The candidate is built in a temp directory and moved into place on success, so a failing
    variant never lands beside the shipped model.
    """
    final_fp16_path = output_path.with_name(output_path.stem + FP16_FILE_SUFFIX + output_path.suffix)
    with tempfile.TemporaryDirectory() as temporary_directory:
        candidate_path = Path(temporary_directory) / final_fp16_path.name
        fp16_size_bytes = save_onnx_model(convert_initializers_to_fp16_with_cast(onnx_model), candidate_path)
        fp16_parity = measure_onnx_parity(model, candidate_path, PARITY_CROP_SIZES)
        worst_fp16_diff = max(max(result.clean_crop_max_abs_diff, result.masks_max_abs_diff) for result in fp16_parity)
        fp16_keeps_parity = worst_fp16_diff <= FP16_PARITY_TOLERANCE
        if fp16_keeps_parity:
            shutil.move(candidate_path, final_fp16_path)
        else:
            logger.warning("fp16 weights diff %.2e > %.2e: not writing %s", worst_fp16_diff, FP16_PARITY_TOLERANCE, final_fp16_path)
    return {
        "onnx_path": str(final_fp16_path) if fp16_keeps_parity else None,
        "file_size_mb": fp16_size_bytes / 1e6,
        "parity": [asdict(result) for result in fp16_parity],
        "keeps_parity": fp16_keeps_parity,
    }
