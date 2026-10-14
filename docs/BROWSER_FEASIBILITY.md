# Browser inference feasibility — work in progress

The current public website remains working and unchanged. **Full Cadence browser inference and numerical parity are not yet demonstrated.** This branch contains a metadata-only sizing prototype and a plan for the runtime and simpler interface; it does not substitute another model or publish mock inference.

## Exact model boundary

The demo pins `foolish-bandit/cadence-0.8b` at `17cdf36034677e406033fbf97637b4db1984ca70`. Its text backbone is Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, reconstructed with the pinned released-Kev adapter before Cadence's own unmerged learned adapter and pointer head.

The cached tensor headers and release configuration agree: 752,393,024 text-backbone parameters plus 5,936,128 learned adapter/head parameters = 758,329,152. The pointer head has two FP32 256×1024 projections and their biases, scoring selected `<decide>` and `</opt>` hidden states with `k(h_options) @ q(h_decide) / 16`. There is no generated chat response or vocabulary readout.

Browser preprocessing must preserve native text/JSON rendering, option order and semantic IDs, `no`/`yes` Boolean option text, zero-based score levels, reserved-token sanitization and all five encoder delimiters. No chat template may replace this encoding. For the one-question demo, the hybrid backbone uses one causal row. Unsupported input must be rejected rather than truncated or silently reformatted.

The published CUDA wrapper autocasts the **entire forward** to BF16. FP32 adapter/head parameter storage therefore does not prove FP32 matrix arithmetic. WGSL exposes f32 and optional f16, not native BF16. A browser FP32 path could be qualified against the separate CPU FP32 reference, with CUDA differences reported; it cannot be labeled CUDA-equivalent without direct evidence. INT8 is not a faithful default: the prior study changed 76/255 paired decisions.

## Current tool support

As inspected on October 14, 2026:

- Current Transformers.js registers Qwen3.5, but its generation wrapper does not implement Cadence's trained pointer task. A stock pipeline is insufficient. [Registry](https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/models/registry.js), [Qwen3.5 class](https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/models/qwen3_5/modeling_qwen3_5.js).
- Optimum-ONNX's inspected model configuration registry has Qwen3/Qwen3-MoE registrations but no Qwen3.5 registration. A custom/direct export is needed. [Export configurations](https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_configs.py).
- ORT-GenAI's Qwen35 builder emits hybrid `CausalConvWithState`/`LinearAttention` operators; its alternate fused GatedDeltaNet path requires CUDA. Those operators are absent from the inspected ORT WebGPU operator table. Actual shipped WASM kernel availability still needs a lightweight probe. [Qwen builder](https://github.com/microsoft/onnxruntime-genai/blob/main/src/python/py/models/builders/qwen.py), [WebGPU operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md).
- ONNX Runtime documents a 2 GB protobuf limit and a 4 GB WASM address space. A large faithful graph needs external-data shards and measured memory, not a promise that a generic conversion fits. [Large-model guidance](https://onnxruntime.ai/docs/tutorials/web/large-models.html).

Practical candidates are custom standard-operator lowering, a separately built WASM runtime, or WebGPU subgraphs joined by a custom DeltaNet kernel. An architecture entry alone does not establish compatibility. Mobile support, actual memory and latency remain unmeasured.

## Size and resource preflight

`python -m browser_export.preflight --package /path/to/pinned/package --base-snapshot /path/to/pinned/base --staging-dir /path/to/staging` reads bounded base safetensors headers, hashes public-package bytes to validate manifest entries and reports counts without importing Torch or deserializing tensors.

The observed storage estimates, before ONNX graph/working copies/runtime assets, are:

| Candidate storage | Weight bytes | Interpretation |
|---|---:|---|
| FP32 reference geometry | 3,033,316,608 | About 3.03 GB / 2.82 GiB; not yet an exported browser artifact |
| 16-bit backbone + FP32 learned tensors | 1,528,530,560 | About 1.53 GB / 1.42 GiB; not a qualified browser precision profile |
| Cached tokenizer JSON/config | 12,823,908 | Additional ~12.8 MB; tokenizer parity remains to be tested |

The host currently has about 2.7 GiB (2.86 GB) free project disk, 16 GiB available RAM and full swap. `/tmp` is a RAM-backed filesystem, so its free space is not independent memory capacity. The NVIDIA driver is unavailable. Full conversion/model inference is held until a bounded compute slot, working-space budget and safe staging location are coordinated. Existing environments were located without importing/loading the model; no new model files or conversion libraries have been downloaded.

**Proposed publishing destination:** a separate public `foolish-bandit/cadence-0.8b-browser` artifact repository with an immutable revision, license notices, source pins, per-file SHA256/bytes and parity evidence. It has not been created or uploaded. Its exact exported size and destination must be reported and approved before any upload outside this demo repository. The published checkpoint stays unchanged.

## Proof gates and simpler interface

1. Match native typed rendering, token IDs, option/decision indices and ordering for frozen choice, JSON Boolean and ordinal cases, including reserved tokens and Unicode. Be explicit about browser numeric/JSON support.
2. Probe actual shipped WASM/WebGPU operators with small synthetic graphs. Test a representative hybrid layer at lengths 63/64/65/128 before a full model export.
3. In a reserved slot, capture native reference logits/probabilities and intermediate precision, then export a bounded single-question graph preserving the exact learned tensors. A successful export is not parity.
4. Compare raw logits, every probability and derived Boolean/score values under declared tolerances; retain errors and precision differences. Measure real browser memory, load time and inference latency before exposing support claims.
5. Build the guided form: **Context → Question → Possible answers**, familiar examples, one obvious action and readable results. Keep semantic IDs/JSON/precision/revision under advanced details. Explain that Cadence identifies requests but does not play music, book tables or fetch live weather.
6. Model preparation requires an explicit **Download model** click showing verified total size and privacy. Provide real byte progress, cancel/retry, complete pinned-cache verification and clear unsupported-device/cache errors. Do not download on page load, invent progress, or fall back to a remote API.
7. Review and test numerical/runtime/download flows and desktop/mobile accessibility, then merge/deploy only after real browser proof. Until then the current site's honest static mode remains intact.

No full model was loaded, converted or benchmarked during this preflight. No browser conversion assets have been published. The unresolved work is a resource-gated custom runtime and parity investigation, not evidence that the existing checkpoint is already browser-compatible.
