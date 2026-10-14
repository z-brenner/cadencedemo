"""Generate tiny, synthetic ONNX graphs; never import Torch or model packages."""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import onnx
from onnx import TensorProto, helper


def generate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    cases = []

    def case(name, op, feeds, outputs, expected=None, domain="", **attrs):
        # None feed entries are positional, omitted optional operator inputs.
        inputs, infos, values = [], [], {}
        for key, spec in feeds:
            inputs.append(key or "")
            if spec is None:
                continue
            dtype, dims, data = spec
            infos.append(helper.make_tensor_value_info(key, dtype, dims))
            values[key] = {
                "type": "int64" if dtype == TensorProto.INT64 else "float32",
                "dims": dims,
                "data": data,
            }
        node = helper.make_node(
            op, inputs, [key for key, _ in outputs], name=name, domain=domain, **attrs
        )
        graph = helper.make_graph(
            [node],
            name,
            infos,
            [
                helper.make_tensor_value_info(key, TensorProto.FLOAT, dims)
                for key, dims in outputs
            ],
        )
        imports = [helper.make_opsetid("", 18)]
        if domain:
            imports.append(helper.make_opsetid(domain, 1))
        model = helper.make_model(
            graph,
            opset_imports=imports,
            ir_version=10,
            producer_name="cadence-synthetic-operator-probe",
        )
        onnx.checker.check_model(model)
        path = destination / f"{name}.onnx"
        onnx.save(model, path)
        cases.append(
            {
                "name": name,
                "operator": f"{domain or 'ai.onnx'}::{op}",
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "feeds": values,
                "expected": expected,
            }
        )

    def f32(dims, data):
        return TensorProto.FLOAT, dims, data

    def i64(dims, data):
        return TensorProto.INT64, dims, data

    case(
        "matmul",
        "MatMul",
        [("a", f32([2, 2], [1, 2, 3, 4])), ("b", f32([2, 2], [1, 0, 0, 1]))],
        [("y", [2, 2])],
        {"y": [1, 2, 3, 4]},
    )
    case(
        "depthwise_conv",
        "Conv",
        [
            ("x", f32([1, 2, 3], [1, 2, 3, 4, 5, 6])),
            ("w", f32([2, 1, 2], [1, 1, 1, 1])),
        ],
        [("y", [1, 2, 3])],
        {"y": [1, 3, 5, 4, 9, 11]},
        group=2,
        pads=[1, 0],
        kernel_shape=[2],
    )
    case(
        "cumsum",
        "CumSum",
        [("x", f32([3], [1, 2, 3])), ("axis", i64([], [0]))],
        [("y", [3])],
        {"y": [1, 3, 6]},
    )
    case(
        "gather",
        "Gather",
        [("x", f32([3, 2], [1, 2, 3, 4, 5, 6])), ("indices", i64([2], [2, 0]))],
        [("y", [2, 2])],
        {"y": [5, 6, 1, 2]},
        axis=0,
    )
    case(
        "reduce_mean",
        "ReduceMean",
        [("x", f32([2, 2], [1, 3, 5, 7])), ("axes", i64([1], [1]))],
        [("y", [2, 1])],
        {"y": [2, 6]},
        keepdims=1,
    )
    case(
        "softmax",
        "Softmax",
        [("x", f32([1, 2], [0, 0]))],
        [("y", [1, 2])],
        {"y": [0.5, 0.5]},
        axis=-1,
    )
    case(
        "exp",
        "Exp",
        [("x", f32([2], [0, 1]))],
        [("y", [2])],
        {"y": [1, 2.718281828459045]},
    )
    case("sqrt", "Sqrt", [("x", f32([2], [1, 4]))], [("y", [2])], {"y": [1, 2]})
    case(
        "scatter_nd",
        "ScatterND",
        [
            ("x", f32([3], [1, 2, 3])),
            ("indices", i64([1, 1], [1])),
            ("updates", f32([1], [9])),
        ],
        [("y", [3])],
        {"y": [1, 9, 3]},
    )
    # Scalar heads simplify the hand-calculated gated-delta recurrence: zero decay,
    # normalized key=1 and beta=.5 yield state/output .5 then 1.25.
    case(
        "linear_attention",
        "LinearAttention",
        [
            ("q", f32([1, 2, 1], [1, 1])),
            ("k", f32([1, 2, 1], [1, 1])),
            ("v", f32([1, 2, 1], [1, 2])),
            ("state", f32([1, 1, 1, 1], [0])),
            ("decay", f32([1, 2, 1], [0, 0])),
            ("beta", f32([1, 2, 1], [0.5, 0.5])),
        ],
        [("y", [1, 2, 1]), ("present", [1, 1, 1, 1])],
        {"y": [0.5, 1.25], "present": [1.25]},
        domain="com.microsoft",
        q_num_heads=1,
        kv_num_heads=1,
        scale=1.0,
        update_rule="gated_delta",
    )
    case(
        "causal_conv_with_state",
        "CausalConvWithState",
        [
            ("x", f32([1, 1, 3], [1, 2, 3])),
            ("w", f32([1, 1, 2], [1, 1])),
            ("b", f32([1], [0])),
            ("state", f32([1, 1, 1], [0])),
        ],
        [("y", [1, 1, 3]), ("present", [1, 1, 1])],
        {"y": [1, 3, 5], "present": [3]},
        domain="com.microsoft",
        activation="none",
    )
    case(
        "gated_delta_net",
        "GatedDeltaNet",
        [
            ("q", f32([1, 2, 1, 1], [1, 1])),
            ("k", f32([1, 2, 1, 1], [1, 1])),
            ("v", f32([1, 2, 1, 1], [1, 2])),
            (None, None),
            ("decay", f32([1, 2, 1], [0, 0])),
            ("beta", f32([1, 2, 1], [0.5, 0.5])),
            ("state", f32([1, 1, 1, 1], [0])),
        ],
        [("y", [1, 2, 1, 1]), ("present", [1, 1, 1, 1])],
        {"y": [0.5, 1.25], "present": [1.25]},
        domain="com.microsoft",
        scale=1.0,
        update_rule="gated_delta",
    )
    case(
        "missing_operator_control",
        "CadenceProbeMissingOperator",
        [("x", f32([1], [1]))],
        [("y", [1])],
        domain="com.microsoft",
    )
    for label, dtype, one in [
        ("bf16", TensorProto.BFLOAT16, 0x3F80),
        ("f16", TensorProto.FLOAT16, 0x3C00),
    ]:
        name = f"{label}_initializer_cast_matmul"
        weight = helper.make_tensor(
            "stored_weight", dtype, [2, 2], struct.pack("<4H", one, 0, 0, one), raw=True
        )
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Cast",
                    ["stored_weight"],
                    ["fp32_weight"],
                    name=name + "_cast",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "MatMul", ["a", "fp32_weight"], ["y"], name=name + "_matmul"
                ),
            ],
            name,
            [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])],
            initializer=[weight],
        )
        model = helper.make_model(
            graph,
            ir_version=10,
            opset_imports=[helper.make_opsetid("", 18)],
            producer_name="cadence-storage-cast-probe",
        )
        onnx.checker.check_model(model)
        path = destination / f"{name}.onnx"
        onnx.save(model, path)
        cases.append(
            {
                "name": name,
                "operator": f"{label.upper()} initializer -> ai.onnx::Cast(FLOAT) -> MatMul",
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "feeds": {
                    "a": {"type": "float32", "dims": [2, 2], "data": [1, 2, 3, 4]}
                },
                "expected": {"y": [1, 2, 3, 4]},
            }
        )
    (destination / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = generate(args.out)
    print(
        json.dumps(
            {"cases": len(cases), "totalGraphBytes": sum(c["bytes"] for c in cases)}
        )
    )
