"""ONNX export of a tiny model: contract signature, opset, WebGPU ops, parity, fp16 weights, no leaked paths."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import helper

from glare_model.architecture.glare_removal_nafnet import GlareRemovalNAFNet
from glare_model.export.onnx_export import (
    convert_initializers_to_fp16_with_cast,
    export_glare_model_to_onnx_model,
    save_onnx_model,
)
from glare_model.export.onnx_verification import measure_onnx_parity
from glare_model.export.opset_downconversion import downconvert_opset18_model_to_opset17
from glare_model.export.webgpu_operator_inventory import (
    UnsupportedWebGPUOperatorError,
    check_graph_uses_only_webgpu_operators,
)

PARITY_SIZES = [(64, 128), (128, 256)]


@pytest.fixture(scope="module")
def exported_tiny_model(tmp_path_factory):
    """Export a tiny randomized model once per module (export is the slow part); returns (model, onnx_model, onnx_path)."""
    torch.manual_seed(0)
    model = GlareRemovalNAFNet(base_width=8, encoder_block_counts=[1, 1, 1, 1], middle_block_count=1, decoder_block_counts=[1, 1, 1, 1])
    for parameter_name, parameter in model.named_parameters():
        if "residual_scale" in parameter_name or "head" in parameter_name:
            parameter.data.normal_(0.0, 0.2)
    onnx_model = export_glare_model_to_onnx_model(model)
    onnx_path = tmp_path_factory.mktemp("export") / "glare_removal.onnx"
    save_onnx_model(onnx_model, onnx_path)
    return model, onnx_model, onnx_path


def test_export_matches_io_contract(exported_tiny_model):
    _, onnx_model, _ = exported_tiny_model
    assert next(entry.version for entry in onnx_model.opset_import if entry.domain == "") == 17
    graph_inputs = {value.name: value for value in onnx_model.graph.input}
    graph_outputs = [value.name for value in onnx_model.graph.output]
    assert list(graph_inputs) == ["glare_crop"]
    assert graph_outputs == ["clean_crop", "masks"]
    input_dims = graph_inputs["glare_crop"].type.tensor_type.shape.dim
    assert [input_dims[0].dim_value, input_dims[1].dim_value] == [1, 3]
    assert [input_dims[2].dim_param, input_dims[3].dim_param] == ["height", "width"]


def test_export_uses_only_webgpu_operators(exported_tiny_model):
    _, onnx_model, _ = exported_tiny_model
    operator_counts = check_graph_uses_only_webgpu_operators(onnx_model)
    assert "GroupNormalization" not in operator_counts and "Constant" not in operator_counts


def test_export_parity_at_two_sizes(exported_tiny_model):
    model, _, onnx_path = exported_tiny_model
    for parity_result in measure_onnx_parity(model, onnx_path, PARITY_SIZES):
        assert parity_result.clean_crop_max_abs_diff < 1e-4
        assert parity_result.masks_max_abs_diff < 1e-4


def test_fp16_weight_variant_keeps_parity_within_one_8bit_level(exported_tiny_model, tmp_path):
    model, onnx_model, _ = exported_tiny_model
    fp16_path = tmp_path / "glare_removal_fp16.onnx"
    save_onnx_model(convert_initializers_to_fp16_with_cast(onnx_model), fp16_path)
    for parity_result in measure_onnx_parity(model, fp16_path, PARITY_SIZES):
        assert max(parity_result.clean_crop_max_abs_diff, parity_result.masks_max_abs_diff) < 1.0 / 255.0


def test_exported_file_carries_no_machine_paths(exported_tiny_model):
    _, _, onnx_path = exported_tiny_model
    file_bytes = Path(onnx_path).read_bytes()
    assert str(Path.home()).encode() not in file_bytes
    assert b"stack_trace" not in file_bytes


def test_inventory_rejects_group_normalization():
    group_norm_graph = helper.make_graph(
        [helper.make_node("GroupNormalization", ["x", "scale", "bias"], ["y"], num_groups=1)],
        "group_norm",
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None) for name in ("x", "scale", "bias")],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
    )
    with pytest.raises(UnsupportedWebGPUOperatorError, match="GroupNormalization"):
        check_graph_uses_only_webgpu_operators(helper.make_model(group_norm_graph))


def test_downconversion_refuses_non_opset18_input():
    empty_graph = helper.make_graph([], "empty", [], [])
    opset17_model = helper.make_model(empty_graph, opset_imports=[helper.make_opsetid("", 17)])
    with pytest.raises(ValueError, match="opset-18"):
        downconvert_opset18_model_to_opset17(opset17_model)


def test_downconversion_moves_reduce_mean_axes_to_attribute():
    axes = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["y"], keepdims=1, noop_with_empty_axes=0)],
        "reduce",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1, 4, 4])],
        initializer=[axes],
    )
    converted = downconvert_opset18_model_to_opset17(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]))
    onnx.checker.check_model(converted, full_check=True)
    reduce_node = converted.graph.node[0]
    assert list(reduce_node.input) == ["x"]
    assert {attribute.name: attribute for attribute in reduce_node.attribute}["axes"].ints == [1]
    assert len(converted.graph.initializer) == 0
