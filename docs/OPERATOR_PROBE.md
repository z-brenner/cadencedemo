# Browser operator and trained-layer probe

ONNX Runtime Web **1.30.0** executes FP32 `com.microsoft::LinearAttention` and `CausalConvWithState` in the actual WASM and native WebGPU backends. A trained Cadence DeltaNet layer also executes with close agreement to its CPU FP32 reference. This is bounded layer evidence, **not a full-model browser export, CUDA parity result, or hardware performance claim**.

## Environment and scope

The probe ran in headless Google Chrome **155.0.8059.39**, controlled by the existing Playwright installation because the Browser skill was absent. ORT tooling was installed only in `/tmp/cadence-browser-probe`; project dependencies were unchanged. WASM used one thread, graph optimization was disabled, and all inputs were local fixtures.

Default Chrome exposes `navigator.gpu` but returns no adapter on this host. A separate session explicitly enabled **SwiftShader software WebGPU**, reporting vendor `google`, architecture `swiftshader`, and `isFallbackAdapter=true`. It does not expose `shader-f16`. Results below distinguish ordinary WebGPU sessions, which permit CPU fallback, from sessions with `session.disable_cpu_ep_fallback="1"`.

Synthetic tests load no trained weights. The trained-layer test loads one already-exported **43,733,539-byte** FP32 graph containing **10,666,400 parameters**, with the unmerged Cadence LoRA branches. Its SHA256 is stored in each case and the source manifest. No full Cadence model was loaded by this browser probe. The parent prepared the native reference and layer export separately; this harness only stages and executes those saved fixtures.

## Actual operator results

| Graph | WASM | SwiftShader WebGPU, CPU fallback disabled |
|---|---|---|
| FP32 LinearAttention, gated-delta update | Executes; hand-calculated output/state match exactly | Executes; output/state match exactly |
| FP32 CausalConvWithState | Executes; output/state match exactly | Executes; output/state match exactly |
| GatedDeltaNet | Session creation fails: schema recognized, implementation missing | Session creation fails: no provider assigned |
| BF16 initializer → Cast(FLOAT) → MatMul | Session creation fails: Cast implementation missing | Session creation fails: no provider assigned to BF16 Cast |
| F16 initializer → Cast(FLOAT) → MatMul | Executes; identity-weight output exact | Session creates; execution fails because this software adapter lacks `shader-f16` |
| MatMul, depthwise Conv, CumSum, Gather, ReduceMean, Softmax, Exp, Sqrt, ScatterND | All nine execute | All nine execute |

The largest standard-operator absolute error was Exp: `8.25484e-8` in WASM and `3.94289e-7` in software WebGPU. Other hand-calculated tests were exact. An invented operator deliberately fails as unregistered. A named MatMul forced onto CPU deliberately fails when CPU fallback is disabled, confirming that the no-fallback setting is enforced.

The older JavaScript [WebGPU operator table](https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/web/docs/webgpu-operators.md) omits the two working hybrid operators. The actual native [WebGPU contrib kernel registry](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc) registers them. The execution evidence supersedes a blanket unsupported claim based on that table. The exact [operator schemas](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/graph/contrib_ops/bert_defs.cc) were used to construct the graphs.

## Qwen-shaped recurrence comparison

Four seeded random fixtures use batch 1, 16 heads, and 128-dimensional keys and values. The reference is Transformers **5.19.0** `torch_chunk_gated_delta_rule`, CPU FP32 without autocast, chunk size 64, two Torch threads, and seed 2357. Query/key vectors use its L2 normalization; query scaling is applied before ORT and ORT `scale=1`. Decay is supplied in **log space** and beta is a sigmoid output. Both the sequence output and final recurrent state are compared.

| Length | WASM output max absolute error | WASM state max absolute error | Strict software WebGPU output error | Strict software WebGPU state error |
|---:|---:|---:|---:|---:|
| 63 | 3.353e-8 | 8.345e-7 | 2.980e-8 | 8.345e-7 |
| 64 | 3.725e-8 | 5.662e-7 | 3.725e-8 | 5.960e-7 |
| 65 | 2.980e-8 | 3.278e-7 | 2.980e-8 | 3.874e-7 |
| 128 | 3.725e-8 | 5.364e-7 | 4.843e-8 | 5.364e-7 |

All output and state comparisons pass the probe's `1e-5` absolute threshold. Raw evidence also reports maximum relative error using denominator `max(abs(reference), 1e-6)` and relative L2 error. The largest relative L2 drift is about `5.47e-7`; maximum per-element relative error reaches about 7.12% for nearly zero state entries despite small absolute errors. This is approximate agreement, not bitwise equality or full-model qualification.

## Actual trained-layer comparison

The parent-exported first DeltaNet module retains its actual base/LoRA parameters, standard Conv, normalization, gates, and a custom lowering to LinearAttention. The saved hidden-state inputs are seeded random tensors, not full-backbone activations. All four WASM and ordinary software WebGPU cases pass against the saved native CPU FP32 layer output.

| Length | WASM max absolute error | Software WebGPU max absolute error | WASM session create / run seconds | Software WebGPU create / run seconds |
|---:|---:|---:|---:|---:|
| 63 | 1.118e-8 | 3.725e-8 | 1.062 / 0.248 | 0.380 / 3.676 |
| 64 | 2.235e-8 | 4.470e-8 | 0.239 / 0.194 | 0.200 / 5.286 |
| 65 | 1.490e-8 | 4.098e-8 | 0.250 / 0.165 | 0.286 / 4.794 |
| 128 | 3.725e-8 | 1.490e-7 | 0.237 / 0.263 | 0.264 / 5.709 |

Session creation includes local graph fetch and initialization. Run timings include `session.run` and its CPU output materialization. They are single measurements on a software GPU under shared host load, and cannot predict real-device or full-model latency. Maximum relative L2 drift is `8.58e-7` in WASM and `1.65e-6` in ordinary software WebGPU.

Strict WebGPU rejects this dynamic trained-layer graph at session creation because some nodes are assigned to CPU. Verbose placement records **110 WebGPU nodes**, including the base/LoRA MatMuls, Conv, LinearAttention and normalization, plus **32 CPU nodes**: four Gather, 23 Unsqueeze and five Concat nodes. ORT explicitly notes that shape-related operations are assigned to CPU for performance. The successful layer therefore uses WebGPU with WASM support for shape plumbing; an entirely GPU-only layer graph has not been demonstrated.

## Evidence and reproduction

- [Default browser and WASM evidence](browser-operator-probe.json)
- [Synthetic software WebGPU and strict no-fallback evidence](browser-operator-probe-software-gpu.json)
- [Random fixture hashes and reference settings](browser-delta-reference.json)
- [Actual trained-layer outputs, errors and timings](browser-trained-layer-probe.json)
- [Verbose provider placement](browser-trained-layer-placement.json)

The tiny graph generator requires ONNX; the random-reference generator additionally requires the pinned Torch/Transformers environment. Model files and fixture binaries remain under `/tmp`, outside Git. From the repository root, after providing those existing Python dependencies:

```sh
npm install --prefix /tmp/cadence-browser-probe --no-audit --no-fund --ignore-scripts onnxruntime-web@1.30.0
python browser_export/operator_probe_models.py --out /tmp/cadence-browser-probe/graphs
python browser_export/operator_probe_delta_reference.py --out /tmp/cadence-browser-probe/graphs
node browser_export/operator_probe.mjs
PROBE_SOFTWARE_GPU=1 PROBE_OUTPUT=/tmp/cadence-browser-probe/operator-evidence-software-gpu.json node browser_export/operator_probe.mjs
```

To use the already-exported trained layer, run `operator_probe_layer_fixtures.py --source /tmp/cadence-trained-layer-probe --out /tmp/cadence-browser-probe/trained-layer-graphs`, then set `PROBE_GRAPHS` to that directory and `PROBE_OUTPUT` to a separate evidence file when running the same browser harness. `PROBE_VERBOSE=1 PROBE_CASE=trained_delta_layer_63` records placement for one fixture. The source export is prepared by the parent's `layer_probe.py`; staging does not load or re-export the full model.

The viable next investigation is a bounded full-row FP32 runtime that preserves the released BF16-rounded base values and exact FP32 learned tensors. BF16 storage cannot currently be expanded using an in-graph BF16 Cast in the tested runtime. External bit unpacking into FP32 is a candidate requiring implementation and memory validation. F16 storage also requires value-level exactness checks and device support; the successful WASM identity fixture does not prove that all checkpoint values survive BF16-to-F16 conversion. Full-model logits/probabilities, full context, download/cache behavior, hardware GPU/mobile support, peak browser memory, and CUDA precision differences remain unverified.
