"""Rewrite the dynamo exporter's opset-18 graph as an equivalent opset-17 graph.

Why this exists: the contract pins opset 17, the dynamo exporter (the only one that emits a
clean graph: no Constant / Identity / Shape / Gather nodes) starts at opset 18, and
`onnx.version_converter` crashes on its ReduceMean nodes ("No initializer or constant input",
onnx 1.23) even though the axes ARE initializers. Between opsets 17 and 18 only two ops in this
network changed signature, so the rewrite is small and explicit:

- ReduceMean-18 takes `axes` as an input; ReduceMean-13 takes it as an attribute and has no
  `noop_with_empty_axes`. We move the constant axes into the attribute.
- Split-18 has a `num_outputs` attribute; Split-13 with no `split` input already splits evenly
  into as many parts as it has outputs, so the attribute is dropped.

Any other op whose schema differs between 17 and 18 makes the pass fail loudly rather than emit
a graph that silently means something else. ORT parity in `onnx_verification.py` re-checks the
numbers after the rewrite.
"""

import logging

import onnx
from onnx import helper, numpy_helper

logger = logging.getLogger(__name__)

SOURCE_OPSET = 18
TARGET_OPSET = 17


def schema_changed_between_opsets(op_type: str) -> bool:
    """True if the default-domain schema of `op_type` differs between opsets 17 and 18."""
    schema_at_target = onnx.defs.get_schema(op_type, TARGET_OPSET)
    schema_at_source = onnx.defs.get_schema(op_type, SOURCE_OPSET)
    return schema_at_target.since_version != schema_at_source.since_version


def downconvert_opset18_model_to_opset17(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy of `model` rewritten from opset 18 to opset 17 (see module docstring).

    Raises:
        ValueError: the graph is not opset 18, or contains an op this pass does not know how to
            rewrite, or a ReduceMean whose axes are not a constant initializer.
    """
    default_domain_version = {entry.domain: entry.version for entry in model.opset_import}.get("")
    if default_domain_version != SOURCE_OPSET:
        raise ValueError(f"expected an opset-{SOURCE_OPSET} model, got opset {default_domain_version}")

    converted_model = onnx.ModelProto()
    converted_model.CopyFrom(model)
    graph = converted_model.graph
    initializer_by_name = {initializer.name: initializer for initializer in graph.initializer}

    for node in graph.node:
        if node.op_type == "ReduceMean":
            rewrite_reduce_mean_axes_input_to_attribute(node, initializer_by_name)
        elif node.op_type == "Split":
            if len(node.input) > 1 and node.input[1]:
                raise ValueError("Split with an explicit split input is not handled by this pass")
            kept_attributes = [attribute for attribute in node.attribute if attribute.name != "num_outputs"]
            del node.attribute[:]
            node.attribute.extend(kept_attributes)
        elif node.domain in ("", "ai.onnx") and schema_changed_between_opsets(node.op_type):
            raise ValueError(f"{node.op_type} changed between opset 17 and 18; add a rewrite for it")

    referenced_names = {input_name for node in graph.node for input_name in node.input}
    unused_initializers = [initializer for initializer in graph.initializer if initializer.name not in referenced_names]
    for initializer in unused_initializers:
        graph.initializer.remove(initializer)

    for opset_entry in converted_model.opset_import:
        if opset_entry.domain == "":
            opset_entry.version = TARGET_OPSET
    logger.info("downconverted opset %d -> %d, dropped %d axes initializers", SOURCE_OPSET, TARGET_OPSET, len(unused_initializers))
    return converted_model


def rewrite_reduce_mean_axes_input_to_attribute(
    node: onnx.NodeProto, initializer_by_name: dict[str, onnx.TensorProto]
) -> None:
    """In place: turn ReduceMean-18 `(data, axes)` into ReduceMean-13 `(data)` with an axes attribute."""
    attribute_by_name = {attribute.name: attribute for attribute in node.attribute}
    noop_attribute = attribute_by_name.get("noop_with_empty_axes")
    if noop_attribute is not None and noop_attribute.i != 0:
        raise ValueError("ReduceMean with noop_with_empty_axes=1 has no opset-17 equivalent")
    kept_attributes = [attribute for attribute in node.attribute if attribute.name != "noop_with_empty_axes"]
    if len(node.input) > 1 and node.input[1]:
        axes_name = node.input[1]
        if axes_name not in initializer_by_name:
            raise ValueError(f"ReduceMean axes {axes_name!r} is not a constant initializer")
        axes = numpy_helper.to_array(initializer_by_name[axes_name]).astype(int).tolist()
        kept_attributes.append(helper.make_attribute("axes", axes))
        del node.input[1:]
    del node.attribute[:]
    node.attribute.extend(kept_attributes)
