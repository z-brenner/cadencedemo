# Evaluation and CPU guidance

Cadence scored **80/256 (31.25%)** on a frozen additional four-task suite. This does not establish broad reasoning superiority. CPU INT8 improved latency on fixed fixtures but changed **76/255 comparable decisions**; the demo therefore keeps plain FP32 as its only exposed CPU profile.

This sanitized summary is derived from the completed `FINAL_GENERALIZATION_CPU_REPORT_v4.md` and experimental CPU artifact README, dated October 14, 2026. Raw inputs, labels, journals and experimental loaders are not included in this repository. These measurements are separate from the demo's [real functionality smokes](VERIFICATION.md), which used Torch 2.11 rather than the study's Torch 2.8.

## Additional generalization evaluation

Each task contributes 64 cases, frozen before forecasts using the first 64 eligible source-input SHA256 values per task. Accuracy is correct top-choice decisions divided by all 256 cases. Chance averages uniform choice, 1/k, over each case's actual options; shapes have 10 options on 55 cases and 11 on nine. Equal-task chance is 19.968%.

| Task | Chance % | Cadence % | Kev % | GLiClass-Instruct % | Laya % |
|---|---:|---:|---:|---:|---:|
| Adapted CRUXEval output | 25.00 | 65.62 | 67.19 | 73.44 | 71.88 |
| Adapted Dyck completion | 25.00 | 28.12 | 26.56 | 39.06 | 34.38 |
| SVG shape recognition | 9.87 | 12.50 | 12.50 | 12.50 | 12.50 |
| Tracking five objects | 20.00 | 18.75 | 21.88 | 23.44 | 18.75 |
| Aggregate (256 cases) | 19.968 | 31.25 | 32.03 | 37.11 | 34.38 |

| Cadence minus comparator | Difference, percentage points | Marginal 95% paired interval |
|---|---:|---:|
| Kev | -0.78 | [-3.52, +1.95] |
| GLiClass-Instruct | -5.86 | [-12.50, +0.78] |
| Laya | -3.12 | [-9.38, +3.12] |

Intervals use 10,000 paired, task-stratified case bootstrap draws with replacement, stdlib MT19937 seed 20261014 and nearest-rank 2.5/97.5% endpoints. They are marginal, not simultaneous intervals. Every aggregate interval includes zero; this establishes neither superiority nor equivalence. Unknown generator/template and deeper source dependence can invalidate case-IID precision. The original supervised-family qualification is a separate, narrower result.

CRUXEval output and Dyck completion were adapted to four choices: same-literal-type source-pool distractors for bounded code outputs, and single-bracket distractors for Dyck. Code was never executed. These are adapted subset scores, not official generation benchmark scores. BBH shapes/tracking retained original options and targets. Sources are [CRUXEval at 190faf16](https://github.com/facebookresearch/cruxeval/tree/190faf16d175b5847b0af05d937872b1fb395942) and [BBH at 9ee07bd4](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/9ee07bd481feebf959a6b59d61ea57bdcf30964d), both with MIT source notices.

All 1,498 candidates were screened against 57 hash-verified prior local training/exposure/evaluation authorities. No exact normalized-state or contiguous 12-token-span matches were found. Task families were absent from available Cadence fine-tuning/qualification manifests. This is a local lexical screen, not proof of semantic independence or absent upstream pretraining/fine-tuning exposure.

All 256 requests fit every native wrapper without truncation. Cadence/Kev used typed pointer queries; GLiClass used its frozen official instruction prefix and original descriptions; Laya used its official SDK and shipped temperatures, including its 11+ option clamp. GPU measurements used an RTX 3050 OEM 8 GiB, batch one: Cadence/Kev BF16 base (Cadence FP32 adapter/head, T=1; Kev shipped temperature), GLiClass FP32/T=1, and Laya BF16 autocast/shipped temperatures. Different interfaces, precision and calibration prevent attributing results solely to architecture. No prompt search, calibration fitting, checkpoint selection or training used this suite.

GLiClass's initial worker expected ID dictionaries but received ordered vectors, causing 256 post-forward validation failures. A representation-only ID bridge was regression-tested; a declared recovery reran five synthetic fixtures and the same 256 cases with prompts, weights, temperatures and ordering unchanged. Those failures were retained, not reported as zero model capability.

## CPU tradeoffs

| CPU profile | Correct / all 256 | Native results | Changed decisions / 255 paired | Maximum probability drift from FP32 |
|---|---:|---:|---:|---:|
| Plain FP32 | 78 / 256 (30.47%) | 255 | reference | reference |
| Merged FP32 | 79 / 256 (30.86%) | 256 | 0 | 0.000007649 |
| Full dynamic INT8 | 80 / 256 (31.25%) | 256 | 76 | 0.795371734 |
| Layerwise dynamic INT8 | 80 / 256 (31.25%) | 256 | 76 | 0.795371734 |

Plain FP32 stopped at a disk guard after 140 saved outcomes. One dispatched case had no durable result and was never replayed; only 115 unattempted cases resumed. That indeterminate case counts incorrect in the 256-case denominator. The apparent 78→79 change for merged FP32 only reflects observing that case, not a quality improvement. Merged FP32 preserved all 255 comparable decisions with drift below 8e-6, but differed from qualified CUDA on 1/256 decisions (maximum drift 0.03653). CPU/CUDA are not bit-identical.

INT8-minus-FP32 accuracy was +0.78 percentage points, paired 95% interval [-3.52, +5.08], with the indeterminate case counted incorrect. No accuracy gain is established. Full and layerwise INT8 produced identical probabilities on all 256 cases plus five separate smoke fixtures. Both quantized 186 backbone Linear layers using Torch x86 dynamic qint8; embeddings, non-Linear operations and pointer head remained FP32. Mean absolute probability drift from plain FP32 was 0.09167. Similar aggregate accuracy does not make INT8 a faithful per-request substitute.

| Fixed synthetic case | Plain FP32 median/p95 ms | Merged FP32 | Full INT8 | Layerwise INT8 |
|---|---:|---:|---:|---:|
| Choice, 2 options | 472.2 / 492.2 | 428.4 / 467.1 | 254.9 / 328.5 | 274.7 / 315.8 |
| Choice, 7 options | 656.8 / 663.8 | 663.8 / 736.5 | 488.9 / 568.6 | 477.1 / 572.6 |
| Boolean, 2 options | 415.2 / 423.3 | 393.4 / 463.5 | 349.4 / 373.0 | 329.1 / 353.1 |
| Score, 6 levels | 595.6 / 609.5 | 628.8 / 711.8 | 373.6 / 483.1 | 391.6 / 546.3 |
| Choice, 150 options | 10175.4 / 11006.1 | 9468.6 / 9998.4 | 6351.4 / 7207.5 | 7718.3 / 8362.0 |

The 150-option research fixture exceeds this demo's 16-option CPU cap. Three warmups and 20 timed requests per case included encode/infer/decode; loading, resource guards and journal writes were excluded. Original 100 timed calls and 15 warmups were preserved, not repeated after interruption. Sequential throughput was 20/sum(times), not concurrent serving capacity. Background load, clock frequency and cache state were uncontrolled; these are observed comparisons, not service SLAs.

| Profile | Observed process peak GiB, including load | Sampled post-load median RSS GiB |
|---|---:|---:|
| Plain FP32 | 5.155 | 3.779 |
| Merged FP32 | 5.117 | 3.851 |
| Full INT8 | 5.119 | 4.304 |
| Layerwise INT8 | 3.901 | 2.490 |

RSS was sampled once per second across synthetic/evaluation inputs, not measured as model-weight-only memory. Full INT8 median speedups were 1.19–1.85×, with almost unchanged peak and increased sampled median RSS. Layerwise INT8 avoided whole-backbone FP32 materialization: 23.8% lower peak and 42.2% lower sampled median RSS than full INT8, but slower on the 150-option fixture. Merged FP32 timing was uneven. Startup timings used inconsistent intervals after interruption and are deliberately omitted from comparisons here.

## Pinned configuration and practical guidance

Public model: [foolish-bandit/cadence-0.8b](https://huggingface.co/foolish-bandit/cadence-0.8b). Study release: `de5009eeca68c08110e12bcd44c08016c5fe871d`. Demo pin: documentation correction `17cdf36034677e406033fbf97637b4db1984ca70`, with unchanged weights/runtime. Dependencies: Qwen/Qwen3.5-0.8B-Base at `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` and jaredpalmer/kev-0.8b at `9a45d25eb2ab761841196625383fa1dff0e56c1e`.

Study host: Linux x86, Intel i5-13400F, 16 logical CPUs, 32 GB RAM; affinity CPUs 0–3, four intra-op/one inter-op threads, OMP/MKL threads four, batch one. Python 3.12.14; Torch 2.8.0+cu128; transformers 5.19.0; peft 0.21.2; safetensors 0.8.0; numpy 2.5.4. The FP32 loader reconstructs the pinned BF16-rounded merged Kev base, then promotes it to FP32, with verified FP32 adaptation/head, T=1, eval/inference_mode and no optimizer. CPU SDPA/reference DeltaNet kernels are unchanged.

For this demo, follow [cache preparation and CPU setup](../README.md#explicit-cpu-fp32-selection). To match study process/thread configuration on a compatible Linux host after preparing caches:

```bash
taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  CADENCE_DEVICE=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python - <<'PY'
import torch
torch.set_num_interop_threads(1)  # once, before constructing workers
import uvicorn
uvicorn.run('backend.app:app', host='127.0.0.1', port=8000, workers=1)
PY
```

The ordinary demo startup fixes four intra-op threads but does not force affinity or inter-op threads. This command matches configuration, not the frozen research experiment: raw cases and the merged/layerwise experimental loaders are not distributed here. CPU-only Torch wheels, other CPUs/OSes and thread counts were not measured. Torch dynamic quantization is deprecated; INT8 results apply to the pinned Torch 2.8 x86 stack. This custom adapter/head is not a generic AutoModel, PEFT-only or GGUF package.

**Keep the demo FP32.** Merged FP32 is a measured faithful optimization candidate with uneven speed; it is not automatically enabled. Neither INT8 route is a faithful default. Probabilities remain model outputs rather than correctness guarantees. Do not tune prompts, temperatures, checkpoints or quantization settings on this now-examined 256-case suite: future development requires independent TRAIN/DEV and a new grouped confirmation holdout.
