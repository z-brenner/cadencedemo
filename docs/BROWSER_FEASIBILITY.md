# Browser inference qualification — local candidate

The complete Cadence model runs in the production guided UI in **Chrome 155 on Linux, using ONNX Runtime Web 1.30.0 WASM**. Model/runtime downloads require an explicit preparation click, and no inference API is called. The default Pages build remains a functional request explorer. External artifact publication and enabling public browser inference remain pending approval and hosting verification.

## Exact model and precision

The model pin remains `foolish-bandit/cadence-0.8b@17cdf36034677e406033fbf97637b4db1984ca70`. Its 752,393,024-parameter text backbone is reconstructed from `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and `jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e`. Cadence adds 5,936,128 trained adapter/head parameters: **758,329,152 total**.

The export retains all 24 text decoder layers, unmerged trained LoRA operations, final normalization and the trained pointer head. It neither exports vision nor adds a vocabulary/chat readout. Pointer scores remain `k(h_options) @ q(h_decide) / 16`.

**Browser arithmetic is CPU-reference FP32.** BF16-representable FP32 weights are stored as exact upper 16 bits, then expanded before ORT loading. Other FP32 weights retain all their bits. Every packed initializer and embedding has a verified bit-identical round trip. This is lossless storage, not FP16 or INT8 arithmetic. Each decoder has an ONNX graph and packed external weights. Thirty-one embedding shards avoid a single 508 MB browser allocation.

DeltaNet's chunk fallback is lowered to ORT's recurrent `com.microsoft::LinearAttention`, changing evaluation order. Comparisons use declared numerical tolerances. The published CUDA wrapper autocasts the entire forward to BF16; **CUDA numerical parity is unverified**. INT8 is not used; prior decision drift is documented in [the CPU guidance](BENCHMARK_AND_CPU_GUIDANCE.md).

## Evidence and limits

- [Tokenizer evidence](../browser_export/tokenizer_probe_parity.json): `@huggingface/tokenizers@0.2.0`, zero dependencies, matches every encoding field in 115/115 Node and real Chrome cases. Delimiters, reserved-token sanitization, Unicode, typed text/JSON rendering, answer order and readout indices match native code. JSON follows the existing canonical browser wire contract. Unsafe integers, nonfinite numbers, invalid Unicode and nesting above 128 levels are rejected.
- [Operator/layer evidence](OPERATOR_PROBE.md): actual WASM and software WebGPU kernels execute the hybrid operators. The legacy WebGPU table was incomplete. BF16 casts inside ONNX are unavailable, motivating external bit unpacking. A trained DeltaNet module passes 63/64/65/128-token comparisons. Mixed WebGPU/WASM works for that module; strict GPU-only placement fails because shape nodes use CPU.
- [Full-stack evidence](browser-full-stack-parity.json): nine CPU fixtures cover text/JSON choice, Boolean and ordinal queries, Unicode/control tokens and exact 511/512-token boundaries. All leading indices agree. Maximum absolute logit drift is **1.717e-5**, probability drift **1.969e-6**. Predeclared gates are `1e-4` logits, `1e-5` probabilities/derived values, and matching leaders. This sample is not a general accuracy or calibration guarantee.
- [Production UI evidence](browser-real-ui-smoke.json): real choice, Boolean and rating inference, explicit download, verified cache reload and deliberate corrupted-cache rejection pass. No API calls, console/page errors, overflow or Axe violations. Desktop 1280×1000 and mobile 390×844 layouts were inspected; mocked UI checks also cover 320px. **Mobile hardware inference is untested.**

Local file-server preparation took 17.00 seconds. Separate UI decisions took 15.97, 12.07 and 13.85 seconds, including cache verification, layer loading and runtime initialization. These are observations on a shared 16-thread, approximately 30.6 GiB Linux host with another CPU job active, not a dedicated benchmark or internet-download prediction. Nine interleaved comparisons plus one graph-loading pass took 90.79 seconds; that is not single-request latency.

The UI smoke sampled **3,558,961,152 bytes** maximum summed RSS across its owned Chrome processes every 500 ms. Shared pages may be counted twice: approximate process memory, not unique physical allocation. Desktop use is recommended. Other browsers/mobile hardware need resource/runtime verification. The default provider is the proven WASM path, with no automatic remote inference or unqualified GPU fallback.

The browser admits **512 packed tokens, 16 answers and one typed question**. Overflow is rejected, never truncated. The original local CPU/CUDA backend remains available when browser assets are not configured.

## Storage and publication

Current staged artifact: **1,542,947,107 bytes including manifest, 90 files**. It includes graphs, losslessly packed weights, pinned tokenizer files, upstream licenses/notices and an export notice explaining that this derived artifact bundles the reconstructed backbone. The manifest records every file's exact bytes and SHA256. The separately built browser runtime adds approximately 28 MB.

**Proposed destination:** public `foolish-bandit/cadence-0.8b-browser`, separate from the unchanged original checkpoint. Nothing has been uploaded or created there; approval is pending. Asset URLs must pin an immutable commit, and the frontend pins the manifest SHA256. Current staged manifest: `3b773382a7463f5b2bffcba7dd0d0896fbccff33e85ed65cedf60fea19529657`. Release metadata changes must refresh that pin and byte count.

Cold preparation makes one manifest request plus 89 file requests. HTTPS GETs omit credentials/referrer and follow public asset redirects. No context, question or answer is sent to the host. Hugging Face and its redirected asset host must permit cross-origin reads. Progress uses manifest sizes, so exposed `Content-Length` is unnecessary. Saved files are reverified; only completed, verified files are reused after cancellation. Missing/corrupt files cannot produce inference. CacheStorage errors are explicit, and storage persistence is not guaranteed. Cancellation terminates the dedicated worker.

Runtime `.mjs`/`.wasm` files come from the demo origin. Single-thread WASM avoids a cross-origin-isolation requirement. Before public enablement, verify actual HF revision/redirect/CORS behavior from Pages, check every hash and repeat a real UI decision against hosted assets.

Latest project disk observation: 1,218,224,128 bytes available, a **324,722,979-byte deficit** for this artifact alone. `/tmp` is RAM-backed. Staging there avoided deleting checkpoints, unrelated Intern exports or caches. Streaming export took 63.88 seconds, peaked near 6.20 GiB RSS and packed one decoder at a time, without a full FP32 duplicate checkpoint. RAM-backed staging is not durable storage.

## Reproduction

`browser_export/preflight.py` reads bounded base tensor headers and hashes package bytes without Torch/deserialization. `reference.py` captures verified offline CPU fixtures. Layer/export/operator/tokenizer scripts, `stack_probe.mjs` and `ui_smoke.mjs` reproduce qualification. Use the pinned native environment and a fresh staging directory; export refuses to overwrite existing assets. Heavy model files do not belong in this Git repository.

Set `VITE_CADENCE_ASSET_BASE` to a trailing-slash asset root and `VITE_CADENCE_MANIFEST_SHA256` to its SHA256 for a local browser-enabled build. With both absent, current API/static Pages modes remain intact. Public replacement awaits publication and hosting checks.
