"""Write the browser's copy of YuNet with a dynamic input size.

The stock `face_detection_yunet_2023mar.onnx` declares its input as [1, 3, 640, 640].
OpenCV's FaceDetectorYN (the Python reference) ignores that and runs the graph at the
photo's own size padded to a multiple of 32; onnxruntime enforces declared dims and
rejects every other size. The graph itself is size-agnostic (Reshapes use -1, Resizes use
scale factors), so this rewrites only the declared input/output dims to symbolic names and
drops the stale intermediate shape annotations. Weights and ops are untouched.

Run from the repo root:
    /Volumes/vega/datasets/glare-off/venv/bin/python -m tests.app.make_yunet_dynamic_input_model
"""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
STOCK_YUNET_PATH = REPO_ROOT / "app" / "models" / "face_detection_yunet_2023mar.onnx"
DYNAMIC_YUNET_PATH = REPO_ROOT / "app" / "models" / "face_detection_yunet_2023mar_dynamic_input.onnx"


def make_dimension_symbolic(dimension: onnx.TensorShapeProto.Dimension, symbol_name: str) -> None:
    """Replace a fixed dim value with a named symbolic dim."""
    dimension.ClearField("dim_value")
    dimension.dim_param = symbol_name


def build_dynamic_input_yunet(stock_model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of YuNet whose input H/W and output anchor counts are symbolic."""
    dynamic_model = onnx.ModelProto()
    dynamic_model.CopyFrom(stock_model)
    input_dims = dynamic_model.graph.input[0].type.tensor_type.shape.dim
    make_dimension_symbolic(input_dims[2], "height")
    make_dimension_symbolic(input_dims[3], "width")
    for graph_output in dynamic_model.graph.output:
        stride_suffix = graph_output.name.split("_")[1]  # cls_8 -> "8"
        make_dimension_symbolic(graph_output.type.tensor_type.shape.dim[1], f"anchors_stride{stride_suffix}")
    del dynamic_model.graph.value_info[:]
    onnx.checker.check_model(dynamic_model)
    return dynamic_model


def check_dynamic_model_matches_stock_at_640(dynamic_model_path: Path) -> None:
    """Both models must give identical outputs at the one size the stock model accepts."""
    rng = np.random.default_rng(0)
    probe_input = (rng.random((1, 3, 640, 640)) * 255).astype(np.float32)
    stock_outputs = onnxruntime.InferenceSession(str(STOCK_YUNET_PATH)).run(None, {"input": probe_input})
    dynamic_outputs = onnxruntime.InferenceSession(str(dynamic_model_path)).run(None, {"input": probe_input})
    # Not bit-equal: onnxruntime picks shape-specialized kernels for the fixed model (~1e-6 drift).
    for stock_output, dynamic_output in zip(stock_outputs, dynamic_outputs):
        if not np.allclose(stock_output, dynamic_output, atol=1e-4):
            raise AssertionError("dynamic-input YuNet differs from stock YuNet at 640x640")
    odd_size_input = probe_input[:, :, :352, :480]
    onnxruntime.InferenceSession(str(dynamic_model_path)).run(None, {"input": np.ascontiguousarray(odd_size_input)})


def main() -> None:
    """Write the dynamic-input model next to the stock one and self-check it."""
    dynamic_model = build_dynamic_input_yunet(onnx.load(str(STOCK_YUNET_PATH)))
    onnx.save(dynamic_model, str(DYNAMIC_YUNET_PATH))
    check_dynamic_model_matches_stock_at_640(DYNAMIC_YUNET_PATH)
    print(f"wrote {DYNAMIC_YUNET_PATH.relative_to(REPO_ROOT)} ({DYNAMIC_YUNET_PATH.stat().st_size} bytes); matches stock at 640x640")


if __name__ == "__main__":
    main()
