"""Export one actual, unmerged Cadence DeltaNet layer to a bounded ONNX probe.

No full-model export. Uses FP32 weights and CPU reference arithmetic. The custom
operator changes chunk evaluation to a recurrent evaluation and must be measured.
"""

import argparse
import hashlib
import json
from pathlib import Path


def make_lowering(heads=16, width=128):
    import torch

    class Recurrent(torch.autograd.Function):
        @staticmethod
        def forward(ctx, query, key, value, decay, beta):
            state = torch.zeros(
                (query.shape[0], heads, width, width), dtype=torch.float32
            )
            outputs = []
            for i in range(query.shape[1]):
                q, k, v = query[:, i], key[:, i], value[:, i]
                state = state * decay[:, i].exp()[..., None, None]
                correction = (v - (state * k[..., None]).sum(-2)) * beta[:, i, :, None]
                state = state + k[..., None] * correction[..., None, :]
                outputs.append((state * q[..., None]).sum(-2))
            return torch.stack(outputs, dim=1)

        @staticmethod
        def symbolic(graph, query, key, value, decay, beta):
            flatten = graph.op(
                "Constant", value_t=torch.tensor([0, 0, -1], dtype=torch.int64)
            )
            q, k, v = [graph.op("Reshape", x, flatten) for x in (query, key, value)]
            state_shape = graph.op(
                "Constant",
                value_t=torch.tensor([1, heads, width, width], dtype=torch.int64),
            )
            state = graph.op(
                "ConstantOfShape", state_shape, value_t=torch.tensor([0.0])
            )
            output, _ = graph.op(
                "com.microsoft::LinearAttention",
                q,
                k,
                v,
                state,
                decay,
                beta,
                q_num_heads_i=heads,
                kv_num_heads_i=heads,
                scale_f=1.0,
                update_rule_s="gated_delta",
                outputs=2,
            )
            shape = graph.op(
                "Constant",
                value_t=torch.tensor([0, 0, heads, width], dtype=torch.int64),
            )
            return graph.op("Reshape", output, shape).setType(query.type())

    def lowered(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **kwargs,
    ):
        if (
            initial_state is not None
            or output_final_state
            or not use_qk_l2norm_in_kernel
        ):
            raise ValueError(
                "Only an uncached one-question row is qualified by this probe."
            )
        query, key, value = [x.float() for x in (query, key, value)]
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
        return Recurrent.apply(
            query / width**0.5, key, value, g.float(), beta.float()
        ), None

    return lowered


def run(destination):
    import numpy as np
    import torch
    from backend.runtime import load_model
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    runtime = load_model(device="cpu")
    layer = next(
        m for m in runtime.model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"
    )
    layer.eval().requires_grad_(False)
    heads, width = layer.num_v_heads, layer.head_v_dim
    if layer.num_k_heads != heads or layer.head_k_dim != width:
        raise ValueError("Probe only supports the pinned equal key/value geometry.")
    native_chunk = qwen.torch_chunk_gated_delta_rule
    lowered = make_lowering(heads, width)

    destination.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    fixtures = []
    for length in (63, 64, 65, 128):
        hidden = torch.randn((1, length, layer.hidden_size), generator=generator) * 0.2
        with torch.inference_mode():
            native = layer(hidden)
            qwen.torch_chunk_gated_delta_rule = lowered
            try:
                candidate = layer(hidden)
            finally:
                qwen.torch_chunk_gated_delta_rule = native_chunk
        np.savez(
            destination / f"layer-{length}.npz",
            hidden=hidden.numpy(),
            expected=native.numpy(),
        )
        fixtures.append(
            {
                "length": length,
                "maxTorchLoweringError": (native - candidate).abs().max().item(),
            }
        )
    qwen.torch_chunk_gated_delta_rule = lowered
    try:
        with torch.inference_mode():
            torch.onnx.export(
                layer,
                (hidden,),
                str(destination / "linear-layer.onnx"),
                input_names=["hidden"],
                output_names=["output"],
                dynamic_axes={"hidden": {1: "sequence"}, "output": {1: "sequence"}},
                opset_version=18,
                custom_opsets={"com.microsoft": 1},
                dynamo=False,
                do_constant_folding=False,
            )
    finally:
        qwen.torch_chunk_gated_delta_rule = native_chunk
    model = destination / "linear-layer.onnx"
    result = {
        "scope": "one actual trained DeltaNet module; not full model",
        "parameters": sum(p.numel() for p in layer.parameters()),
        "modelBytes": model.stat().st_size,
        "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "precision": "FP32",
        "fixtures": fixtures,
        "browserTested": False,
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
