"""Stream exact FP32 Cadence graphs into losslessly packed browser layer assets.

FP32 tensors whose lower 16 bits are zero are stored as BF16 bits, with no
arithmetic/weight rounding. Browser code must unpack them before ORT loading.
Each decoder graph is exported and packed independently to bound working copies.
This is an experimental export; native/browser parity is a separate gate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from backend.config import MODEL_ID, REVISION
from browser_export.layer_probe import make_lowering


def pack_graph(source, destination, name):
    import numpy as np
    import onnx
    from onnx import external_data_helper

    model = onnx.load(source)
    pieces, offset = [], 0
    packed = destination / f"{name}.weights"
    with packed.open("wb") as stream:
        for tensor in model.graph.initializer:
            if tensor.data_type != onnx.TensorProto.FLOAT or not tensor.raw_data:
                continue
            raw = tensor.raw_data
            bits = np.frombuffer(raw, dtype="<u4")
            bf16 = not np.any(bits & 65535)
            data = (bits >> 16).astype("<u2").tobytes() if bf16 else raw
            # Lossless bit equality is mandatory for every serialized initializer.
            restored = (
                (np.frombuffer(data, dtype="<u2").astype("<u4") << 16)
                if bf16
                else np.frombuffer(data, dtype="<u4")
            )
            if not np.array_equal(bits, restored):
                raise ValueError("Packed weights do not preserve FP32 bits.")
            begin = stream.tell()
            stream.write(data)
            pieces.append(
                {
                    "offset": begin,
                    "bytes": len(data),
                    "decodedOffset": offset,
                    "decodedBytes": len(raw),
                    "encoding": "bf16-bits" if bf16 else "fp32-bits",
                }
            )
            external_data_helper.set_external_data(
                tensor, location=f"{name}.fp32", offset=offset, length=len(raw)
            )
            tensor.ClearField("raw_data")
            offset += len(raw)
    graph = destination / f"{name}.onnx"
    onnx.save(model, graph)
    source.unlink()  # Only this task's transient unpacked export; no source checkpoint.
    return {
        "name": name,
        "graph": graph.name,
        "weights": packed.name,
        "externalPath": f"{name}.fp32",
        "decodedBytes": offset,
        "packing": pieces,
        "inputs": [i.name for i in model.graph.input],
    }


def export(destination, max_tokens=512):
    import numpy as np
    import torch
    from backend.runtime import load_model
    from browser_export.reference import capture
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    if destination.exists():
        raise ValueError(
            "Choose a fresh staging directory; existing assets are never overwritten."
        )
    if max_tokens != 512:
        raise ValueError("Initial browser qualification cap is exactly 512 tokens.")
    destination.mkdir(parents=True)
    runtime = load_model(device="cpu")
    reference = capture(fp16_probe=True, runtime=runtime)
    Path("docs/browser-native-reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n"
    )
    text = next(
        m for m in runtime.model.modules() if type(m).__name__ == "Qwen3_5TextModel"
    )
    # Explicit attention path uses exportable, ordinary MatMul/Softmax operations.
    text.config._attn_implementation = "eager"
    hidden = text.embed_tokens(torch.tensor([reference["cases"][0]["encoding"]["ids"]]))
    seq = hidden.shape[1]
    all_positions = torch.arange(max_tokens).reshape(1, -1)
    with torch.inference_mode():
        cos_all, sin_all = text.rotary_emb(hidden, all_positions)
    np.concatenate([cos_all.numpy(), sin_all.numpy()], axis=0).astype("<f4").tofile(
        destination / "rotary.fp32"
    )
    cos, sin = cos_all[:, :seq], sin_all[:, :seq]
    mask = torch.full((1, 1, seq, seq), torch.finfo(torch.float32).min).triu(1)

    class Decoder(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden, cos, sin, mask):
            return self.layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=mask
                if self.layer.block_type == "full_attention"
                else None,
                past_key_values=None,
            )

    class Readout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = text.norm
            self.head = runtime.model.head

        def forward(self, hidden, options, decide):
            normalized = self.norm(hidden)[0]
            return self.head(normalized[decide][0], normalized[options])

    graphs = []
    native_chunk = qwen.torch_chunk_gated_delta_rule
    qwen.torch_chunk_gated_delta_rule = make_lowering()
    try:
        for index, layer in enumerate(text.layers):
            name = f"layer-{index:02}"
            temporary = destination / f"{name}.unpacked.onnx"
            with torch.inference_mode():
                torch.onnx.export(
                    Decoder(layer),
                    (hidden, cos, sin, mask),
                    str(temporary),
                    input_names=["hidden", "cos", "sin", "mask"],
                    output_names=["output"],
                    dynamic_axes={
                        "hidden": {1: "sequence"},
                        "cos": {1: "sequence"},
                        "sin": {1: "sequence"},
                        "mask": {2: "sequence", 3: "sequence"},
                        "output": {1: "sequence"},
                    },
                    opset_version=18,
                    custom_opsets={"com.microsoft": 1},
                    dynamo=False,
                    do_constant_folding=False,
                )
            graphs.append(pack_graph(temporary, destination, name))
            print(
                json.dumps(
                    {
                        "exportedLayer": index,
                        "packedBytes": (destination / f"{name}.weights").stat().st_size,
                    }
                ),
                flush=True,
            )
        name = "readout"
        temporary = destination / "readout.unpacked.onnx"
        enc = reference["cases"][0]["encoding"]
        with torch.inference_mode():
            torch.onnx.export(
                Readout(),
                (
                    hidden,
                    torch.tensor(enc["opt_idx"][0]),
                    torch.tensor(enc["decide_idx"]),
                ),
                str(temporary),
                input_names=["hidden", "options", "decide"],
                output_names=["logits"],
                dynamic_axes={
                    "hidden": {1: "sequence"},
                    "options": {0: "answers"},
                    "logits": {0: "answers"},
                },
                opset_version=18,
                dynamo=False,
                do_constant_folding=False,
            )
        graphs.append(pack_graph(temporary, destination, name))
    finally:
        qwen.torch_chunk_gated_delta_rule = native_chunk

    embedding = text.embed_tokens.weight.detach().numpy()
    embedding_bits = embedding.view("<u4")
    if np.any(embedding_bits & 65535):
        raise ValueError("The frozen embedding is not exactly BF16-representable.")
    embedding_shards = []
    for start in range(0, embedding.shape[0], 8192):
        bits = embedding_bits[start : start + 8192]
        packed_embedding = (bits >> 16).astype("<u2")
        if not np.array_equal(packed_embedding.astype("<u4") << 16, bits):
            raise ValueError("Embedding packing lost bits.")
        filename = f"embedding-{len(embedding_shards):02}.bf16"
        packed_embedding.tofile(destination / filename)
        embedding_shards.append(
            {"file": filename, "rowStart": start, "rows": len(bits)}
        )
    # Resolve only the already-cached, pinned tokenizer; never ask the network.
    from huggingface_hub import snapshot_download

    base = runtime.config["base_model"]
    tokenizer_path = Path(
        snapshot_download(
            base["repo"],
            revision=base["revision"],
            allow_patterns=["tokenizer.json", "tokenizer_config.json"],
            local_files_only=True,
            token=False,
        )
    )
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(tokenizer_path / filename, destination / filename)
    for filename in ("LICENSE", "NOTICE", "LICENSE.QWEN", "NOTICE.QWEN"):
        shutil.copyfile(runtime.path / filename, destination / filename)
    (destination / "EXPORT_NOTICE").write_text(
        "Cadence browser export\n\n"
        "NOTICE and NOTICE.QWEN are preserved verbatim from the pinned original "
        "adapter distribution. Their statements that dependencies are not bundled "
        "describe that original distribution. This derived browser artifact DOES "
        "include the reconstructed BF16-rounded merged Kev/Qwen text backbone, "
        "the unmerged Cadence adapter, and the trained pointer head.\n\n"
        "All trained FP32 weight bits are preserved; BF16-representable tensors "
        "are stored as BF16 bits and expanded before FP32 inference. ONNX "
        "DeltaNet recurrence changes evaluation order. Qualification is against "
        "the separately documented CPU FP32 reference, not the CUDA BF16 runtime. "
        "No upstream endorsement is implied.\n"
    )

    manifest = {
        "format": "cadence-browser-layer-stack-v1",
        "model": MODEL_ID,
        "revision": REVISION,
        "baseModel": base,
        "releasedKev": runtime.config["released_kev"],
        "profile": "cpu-fp32-reference; BF16 bits unpacked losslessly into FP32 weights",
        "arithmetic": "FP32; recurrent DeltaNet lowering",
        "browserParityVerified": False,
        "cudaParityVerified": False,
        "maxTokens": max_tokens,
        "hiddenSize": int(text.config.hidden_size),
        "vocabSize": int(embedding.shape[0]),
        "rotaryWidth": int(cos_all.shape[-1]),
        "embeddingShards": embedding_shards,
        "rotary": "rotary.fp32",
        "graphs": graphs,
        "files": {},
    }
    for path in sorted(destination.iterdir()):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        }
    manifest["assetBytesExcludingManifest"] = sum(
        x["bytes"] for x in manifest["files"].values()
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "assetBytesExcludingManifest": manifest["assetBytesExcludingManifest"],
                "manifestBytes": (destination / "manifest.json").stat().st_size,
                "browserParityVerified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    export(parser.parse_args().out)
