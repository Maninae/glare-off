"""Check an exported graph against onnxruntime-web's WebGPU operator table.

The table below is every operator name listed in
https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md at upstream
commit 3eda9022 (2026-07-29), the doc for onnxruntime-web 1.30. An op outside it silently falls
back to CPU in the browser (or fails), so export fails loudly instead.

- Some listed ops carry caveats in that doc: Shape and Reshape have "no GPU kernel", Resize has
  no align_corners when downsampling. `WEBGPU_OPERATORS_WITH_CAVEATS` flags them so
  the inventory report shows them even though they are allowed.
- GroupNormalization, Max, Min, Constant, and Identity are NOT in the table.
"""

import collections

import onnx

ORT_WEBGPU_OPERATOR_TABLE_SOURCE = (
    "onnxruntime js/web/docs/webgpu-operators.md @ 3eda9022d9a57aea63a44d7266252b40fe232e9c (2026-07-29)"
)

WEBGPU_SUPPORTED_OPERATORS = frozenset(
    {
        "Abs", "Acos", "Acosh", "Add", "ArgMax", "ArgMin", "Asin", "Asinh", "Atan", "Atanh",
        "Attention", "AveragePool", "BatchNormalization", "BiasAdd", "BiasSplitGelu", "Cast", "Ceil",
        "Clip", "Concat", "Conv", "ConvTranspose", "Cos", "Cosh", "CumSum", "DFT", "DepthToSpace",
        "DequantizeLinear", "Div", "Einsum", "Elu", "Equal", "Erf", "Exp", "Expand", "FastGelu",
        "Flatten", "Floor", "FusedConv", "Gather", "GatherBlockQuantized", "GatherElements",
        "GatherND", "Gelu", "Gemm", "GlobalAveragePool", "GlobalMaxPool", "Greater",
        "GreaterOrEqual", "GridSample", "GroupQueryAttention", "HardSigmoid", "HardSwish", "If",
        "InstanceNormalization", "LayerNormalization", "LeakyRelu", "Less", "LessOrEqual", "Log",
        "MatMul", "MatMulNBits", "MaxPool", "MemcpyFromHost", "MemcpyToHost", "Mul",
        "MultiHeadAttention", "Neg", "Not", "Pad", "Pow", "QuickGelu", "Range", "Reciprocal",
        "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
        "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "Resize",
        "RotaryEmbedding", "ScatterND", "Shape", "Sigmoid", "SimplifiedLayerNormalization", "Sin",
        "Sinh", "SkipLayerNormalization", "SkipSimplifiedLayerNormalization", "Slice", "Softmax",
        "Split", "Sqrt", "Squeeze", "Sub", "Tan", "Tanh", "ThresholdedRelu", "Tile", "Transpose",
        "Unsqueeze", "Where",
    }
)

WEBGPU_OPERATORS_WITH_CAVEATS = {
    "Shape": "no GPU kernel (runs on CPU)",
    "Reshape": "no GPU kernel (runs on CPU)",
    "Resize": "align_corners unsupported when downsampling",
}


class UnsupportedWebGPUOperatorError(RuntimeError):
    """Raised when an exported graph contains an op outside the WebGPU operator table."""


def count_graph_operators(model: onnx.ModelProto) -> dict[str, int]:
    """Return {op_type: count} over the top-level graph nodes, most frequent first."""
    operator_counts = collections.Counter(node.op_type for node in model.graph.node)
    return dict(operator_counts.most_common())


def check_graph_uses_only_webgpu_operators(model: onnx.ModelProto) -> dict[str, int]:
    """Return the op inventory, raising if any op is missing from the WebGPU table.

    Raises:
        UnsupportedWebGPUOperatorError: lists every offending op and its count.
    """
    operator_counts = count_graph_operators(model)
    unsupported = {op: count for op, count in operator_counts.items() if op not in WEBGPU_SUPPORTED_OPERATORS}
    if unsupported:
        raise UnsupportedWebGPUOperatorError(
            f"ops not in onnxruntime-web WebGPU table ({ORT_WEBGPU_OPERATOR_TABLE_SOURCE}): {unsupported}"
        )
    return operator_counts


def describe_operator_caveats(operator_counts: dict[str, int]) -> list[str]:
    """Return one human-readable line per used op that carries a caveat in the WebGPU doc."""
    return [f"{op}: {WEBGPU_OPERATORS_WITH_CAVEATS[op]}" for op in operator_counts if op in WEBGPU_OPERATORS_WITH_CAVEATS]
