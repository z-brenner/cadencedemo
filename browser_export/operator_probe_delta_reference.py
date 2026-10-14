"""Bounded random-tensor CPU reference; no Cadence/base weights are loaded."""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper
import torch
import transformers
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    l2norm,
    torch_chunk_gated_delta_rule,
)


def generate(destination):
    if transformers.__version__ != "5.19.0":
        raise RuntimeError("Require pinned Transformers 5.19.0 reference")
    torch.set_num_threads(2)
    torch.manual_seed(2357)
    cases = json.loads((destination / "cases.json").read_text())
    cases = [c for c in cases if not c["name"].startswith("delta_chunk_")]
    reference = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "precision": "CPU float32; no autocast",
        "threads": 2,
        "seed": 2357,
        "batch": 1,
        "heads": 16,
        "keyDim": 128,
        "valueDim": 128,
        "chunkSize": 64,
        "relativeErrorDenominatorFloor": 1e-6,
        "cases": [],
    }
    for length in (63, 64, 65, 128):
        name = f"delta_chunk_{length}"
        shape = (1, length, 16, 128)
        q, k, v = [torch.randn(shape, dtype=torch.float32) for _ in range(3)]
        a = torch.randn((1, length, 16), dtype=torch.float32) * 0.2
        a_log = torch.full((16,), -1.0)
        dt_bias = torch.zeros(16)
        decay = -a_log.exp() * torch.nn.functional.softplus(a + dt_bias)
        beta = torch.randn((1, length, 16), dtype=torch.float32).sigmoid()
        with torch.inference_mode():
            expected, final_state = torch_chunk_gated_delta_rule(
                q,
                k,
                v,
                decay,
                beta,
                chunk_size=64,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        packed_shape = [1, length, 16 * 128]
        arrays = {
            "q": l2norm(q) / (128**0.5),
            "k": l2norm(k),
            "v": v,
            "state": torch.zeros((1, 16, 128, 128)),
            "decay": decay,
            "beta": beta,
        }
        feeds = {}
        infos = []
        for key, value in arrays.items():
            dims = packed_shape if key in ("q", "k", "v") else list(value.shape)
            filename = f"{name}-{key}.f32"
            value.detach().contiguous().numpy().astype("<f4").tofile(
                destination / filename
            )
            feeds[key] = {"type": "float32", "dims": dims, "dataFile": filename}
            infos.append(helper.make_tensor_value_info(key, TensorProto.FLOAT, dims))
        expected_files = {}
        for key, value in {"y": expected, "present": final_state}.items():
            filename = f"{name}-expected-{key}.f32"
            value.detach().contiguous().numpy().astype("<f4").tofile(
                destination / filename
            )
            expected_files[key] = filename
        node = helper.make_node(
            "LinearAttention",
            list(feeds),
            ["y", "present"],
            name=name,
            domain="com.microsoft",
            q_num_heads=16,
            kv_num_heads=16,
            scale=1.0,
            update_rule="gated_delta",
            chunk_size=64,
        )
        graph = helper.make_graph(
            [node],
            name,
            infos,
            [
                helper.make_tensor_value_info("y", TensorProto.FLOAT, packed_shape),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, [1, 16, 128, 128]
                ),
            ],
        )
        model = helper.make_model(
            graph,
            ir_version=10,
            producer_name="cadence-random-delta-probe",
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("com.microsoft", 1),
            ],
        )
        onnx.checker.check_model(model)
        model_path = destination / f"{name}.onnx"
        onnx.save(model, model_path)
        cases.append(
            {
                "name": name,
                "operator": "com.microsoft::LinearAttention",
                "file": model_path.name,
                "bytes": model_path.stat().st_size,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "feeds": feeds,
                "expected": None,
                "expectedFiles": expected_files,
            }
        )
        files = {
            file: {
                "bytes": (destination / file).stat().st_size,
                "sha256": hashlib.sha256((destination / file).read_bytes()).hexdigest(),
            }
            for file in [
                model_path.name,
                *[f["dataFile"] for f in feeds.values()],
                *expected_files.values(),
            ]
        }
        reference["cases"].append({"length": length, "files": files})
    (destination / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (destination / "delta-reference.json").write_text(
        json.dumps(reference, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "cases": 4,
                "reference": str(destination / "delta-reference.json"),
                "syntheticDataBytes": sum(
                    f["bytes"] for c in reference["cases"] for f in c["files"].values()
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    generate(parser.parse_args().out)
