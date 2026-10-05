"""Export GlareRemovalNAFNet to the contract ONNX graph, plus an fp16-weights variant.

Contract (repo CLAUDE.md, "Model I/O"): opset 17, input `glare_crop` [1, 3, H, W] with dynamic
H and W (multiples of 16), outputs `clean_crop` [1, 3, H, W] and `masks` [1, 2, H, W].

Pipeline: torch dynamo exporter at opset 18 -> `downconvert_opset18_model_to_opset17` ->
`onnx.checker` -> WebGPU op inventory. The batch dimension is exported as a static 1 because
the browser runs one crop at a time and a static batch keeps shape logic out of the graph.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet
from glare_model.export.opset_downconversion import SOURCE_OPSET, downconvert_opset18_model_to_opset17
from glare_model.export.webgpu_operator_inventory import check_graph_uses_only_webgpu_operators

logger = logging.getLogger(__name__)

INPUT_NAME = "glare_crop"
OUTPUT_NAMES = ("clean_crop", "masks")
EXPORT_SAMPLE_HEIGHT = 256
EXPORT_SAMPLE_WIDTH = 512
# Upper bound for the dynamic size, in units of 16 px (256 * 16 = 4096 px per side).
MAX_SIZE_IN_MULTIPLES = 256
FP16_INITIALIZER_SUFFIX = "__fp16"


def export_glare_model_to_onnx_model(model: GlareRemovalNAFNet) -> onnx.ModelProto:
    """Export `model` (any device; a CPU eval copy is exported, the caller's model is untouched) to a checked opset-17 ModelProto.

    Raises:
        UnsupportedWebGPUOperatorError: if the graph contains an op outside the WebGPU table.
    """
    export_model = copy.deepcopy(model).to("cpu").eval()
    sample_glare_crop = torch.rand(1, 3, EXPORT_SAMPLE_HEIGHT, EXPORT_SAMPLE_WIDTH)
    size_multiple = export_model.required_size_multiple
    height_units = torch.export.Dim("height_units", min=1, max=MAX_SIZE_IN_MULTIPLES)
    width_units = torch.export.Dim("width_units", min=1, max=MAX_SIZE_IN_MULTIPLES)
    with torch.no_grad():
        export_program = torch.onnx.export(
            export_model,
            (sample_glare_crop,),
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            opset_version=SOURCE_OPSET,
            dynamo=True,
            dynamic_shapes={INPUT_NAME: {2: size_multiple * height_units, 3: size_multiple * width_units}},
            verbose=False,
        )
    opset18_model = export_program.model_proto
    opset17_model = downconvert_opset18_model_to_opset17(opset18_model)
    strip_exporter_debug_metadata(opset17_model)
    rename_dynamic_dimensions(opset17_model)
    onnx.checker.check_model(opset17_model, full_check=True)
    check_graph_uses_only_webgpu_operators(opset17_model)
    return opset17_model


def strip_exporter_debug_metadata(model: onnx.ModelProto) -> None:
    """In place: drop per-node metadata, intermediate value_info, and doc strings.

    The dynamo exporter attaches the python stack trace and FX node text to every node, which
    embeds absolute paths of the exporting machine (a privacy leak in a public repo) and adds
    ~1 MB. Intermediate shapes are re-inferred by onnxruntime; graph input/output shapes stay.
    """
    for node in model.graph.node:
        del node.metadata_props[:]
        node.doc_string = ""
    del model.graph.value_info[:]
    del model.metadata_props[:]
    model.doc_string = ""
    model.graph.doc_string = ""


def rename_dynamic_dimensions(model: onnx.ModelProto) -> None:
    """In place: name the dynamic spatial dims `height` / `width` on every graph input and output.

    The dynamo exporter names them after the symbolic expression (`16*height_units`), which is
    noise to anyone reading the model in Netron or the app's console.
    """
    for value_info in list(model.graph.input) + list(model.graph.output):
        dimensions = value_info.type.tensor_type.shape.dim
        for dimension_index, dimension_name in ((2, "height"), (3, "width")):
            if dimensions[dimension_index].dim_param:
                dimensions[dimension_index].dim_param = dimension_name


def convert_initializers_to_fp16_with_cast(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy storing every float32 initializer as float16, cast back to float32 at load.

    Halves the download while compute stays fp32 on every execution provider: ORT constant-folds
    the Cast nodes when the session is created, so no fp16 kernel is ever needed (ORT's CPU/WASM
    EP has almost none).
    """
    fp16_model = onnx.ModelProto()
    fp16_model.CopyFrom(model)
    graph = fp16_model.graph
    cast_nodes = []
    converted_initializers = []
    for initializer in list(graph.initializer):
        if initializer.data_type != onnx.TensorProto.FLOAT:
            continue
        fp16_values = numpy_helper.to_array(initializer).astype(np.float16)
        fp16_name = initializer.name + FP16_INITIALIZER_SUFFIX
        converted_initializers.append(numpy_helper.from_array(fp16_values, name=fp16_name))
        cast_nodes.append(helper.make_node("Cast", [fp16_name], [initializer.name], to=onnx.TensorProto.FLOAT))
        graph.initializer.remove(initializer)
    graph.initializer.extend(converted_initializers)
    original_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(cast_nodes + original_nodes)
    onnx.checker.check_model(fp16_model, full_check=True)
    return fp16_model


def save_onnx_model(model: onnx.ModelProto, output_path: Path) -> int:
    """Write `model` as a single self-contained file (no external data); return its size in bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    file_size_bytes = output_path.stat().st_size
    logger.info("wrote %s (%.2f MB)", output_path, file_size_bytes / 1e6)
    return file_size_bytes
