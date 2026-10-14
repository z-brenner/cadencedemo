# Cadence demo

[Open the public request explorer](https://z-brenner.github.io/cadencedemo/) · [Hosting and runtime boundaries](docs/HOSTING.md)

A local playground for [Cadence 0.8B](https://huggingface.co/foolish-bandit/cadence-0.8b): give it context, a question and criteria, then inspect a typed decision distribution. Cadence is a ~758M parameter decision model. It **does not generate chat or text**.

- **Choice** — labeled options, returned by their semantic IDs.
- **Boolean** — native `noul` query; the answer is P(true).
- **Score** — ordered level descriptions; the answer is the probability-weighted mean of **zero-based** level indices. Five levels span 0–4, not 1–5.

Probabilities are model outputs, **not calibrated guarantees of correctness**. Examples illustrate the interface; they are not a generalization benchmark. The published qualification covers training-seen task families. The fresh additional four-task evaluation scored 80/256 (31.25%) and does not establish broad reasoning superiority; upstream unseen status remains unknown. See [benchmark and CPU guidance](docs/BENCHMARK_AND_CPU_GUIDANCE.md) for definitions, comparisons and caveats. Long-context quality is not established.

## Verification status

The local backend uses real inference only. Mocked responses exist solely in automated tests, and screenshot previews label them explicitly. **Real CUDA and CPU FP32 end-to-end inference passed** for choice, JSON-state Boolean and ordinal score HTTP requests. CUDA also rendered a real decision in the built UI; CPU rendered all three query types in the built UI without API interception. Both cached smokes used **Torch 2.11.0+cu128**, separately recorded from the published Torch 2.8 profile. These are bounded functionality checks, not accuracy, numerical-equivalence or performance benchmarks. Passing stub tests alone are not native inference evidence. See [verification evidence](docs/VERIFICATION.md).

## Public website

GitHub Pages serves a static request explorer: edit the six examples, validate choice/Boolean/score inputs, inspect and copy their exact JSON. It produces no model decisions, probabilities or mocked inference, and sends no input to a backend. To run the model, follow local setup and open the local playground in a separate tab. Pages does not connect to localhost; the loopback API has no cross-origin CORS permission. No hosted inference backend or credentials are configured. See [hosting details](docs/HOSTING.md).

## Local setup

Requirements: Node.js 22.12+, Python **3.12**, and access to Hugging Face when preparing the model cache. Default `CADENCE_DEVICE=cuda` uses the public CUDA/BF16 profile and requires compatible NVIDIA hardware/driver. Explicit `CADENCE_DEVICE=cpu` uses the coordinated experimental FP32 port described below. MPS and automatic device fallback are unsupported.

Clone the repository, then create an isolated environment:

```bash
git clone https://github.com/z-brenner/cadencedemo.git
cd cadencedemo
```

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r backend/requirements.txt -r backend/requirements-model.txt
npm ci
```

The setup dependencies match the published CUDA profile. The coordinated CPU port used a CUDA-enabled Torch wheel while executing on CPU; other wheel/CPU/OS combinations are not qualified here. CUDA driver/wheel compatibility is a local requirement. The demo pins the public package at **`17cdf36034677e406033fbf97637b4db1984ca70`**, the documentation correction after release `de5009eeca68c08110e12bcd44c08016c5fe871d`; adapter/head and runtime code are unchanged. Source/config/learned-weight SHA256 hashes are checked against [the manifest](backend/model-manifest.json), including when using a local package directory. The upstream loader also verifies its pinned dependencies and tensor inventory.

Start two terminals in this repository:

```bash
# Terminal 1: activate .venv, then run ONE worker. Do not use --reload with a loaded GPU model.
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to the backend. Pick an example, edit its state/instructions/criteria and click **Run decision**. The first valid CUDA request downloads and loads the pinned model; allow several minutes. CPU is cached-only and reconstructs its FP32 model on first use. Later requests reuse it. Starting either server alone does not load the model or occupy the GPU.

Fresh use downloads approximately **1.8 GB** of upstream artifacts, plus the ~24 MB Cadence package. The model card reports observed CUDA allocation near 1.6 GiB on its test hardware; this is not a memory requirement guarantee. Inputs and surrounding GPU workloads affect memory and latency. Coordinate shared GPU use before starting inference.

The UI runs locally and sends input only to your backend. Model setup contacts Hugging Face. There is no analytics, request-history database or browser persistence. Errors omit input text; backend logs include diagnostic exceptions. The result's `seconds` measures the model call including encoding and CPU transfer; displayed timing excludes loading, HTTP overhead and UI rendering.

### Offline or existing local package

Prepare the pinned snapshots and upstream dependencies before going offline. For a local copy of the exact public package:

```bash
export CADENCE_MODEL_DIR=/absolute/path/to/pinned/package
export HF_HUB_OFFLINE=1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The package directory must match every manifest hash; arbitrary or modified runtimes are rejected before import. Only the five manifest-listed model Python modules are imported through an isolated source loader; the model directory never receives import precedence. Extra modules/packages or bytecode cannot supply model dependencies. The loader compiles the exact bytes it rechecks against the manifest. Cached model modules must belong to that loader; source/alias dependencies from the model directory are rejected. Do not place the model directory on PYTHONPATH or sys.path; use CADENCE_MODEL_DIR alone. `HF_HUB_OFFLINE=1` also stops upstream Transformers/Hub downloads. Default use downloads the pinned public package through the normal Hugging Face cache. No token is required for the public package. Upstream artifacts must already be cached for offline use.

### Explicit CPU FP32 selection

The demo includes an unchanged, SHA-checked copy of the coordinated `CpuCadence` adapter. It reconstructs the pinned BF16-rounded merged Kev base and then promotes the model to FP32, retaining the FP32 adapter/head. This is a **separate experimental precision profile**, not a claim of CUDA numerical equivalence or general task accuracy. The coordinator has run synthetic reconstruction checks on Linux x86; this demo has tested adapter selection, metadata, limits and failure handling, and **passed cached end-to-end CPU FP32 HTTP/UI smoke** for all three native query types on the tested Linux host. Merged and INT8 profiles are not exposed or recommended here.

Prepare the exact public package, released Kev dependency and base snapshots while network access is available. This cache preparation downloads files without loading Torch or running inference; allow roughly 1.8 GB of downloads and enough installation/cache disk space:

```bash
python - <<'PY'
import json
from huggingface_hub import snapshot_download
from backend.runtime import verified_package
package = verified_package()
cfg = json.loads((package / 'cadence_config.json').read_text())
kev = cfg['released_kev']
cache = package.parent / 'public_cache'
snapshot_download(kev['repo'], revision=kev['revision'],
                  allow_patterns=list(kev['required_files']), token=False,
                  cache_dir=str(cache) if cache.is_dir() else None)
base = cfg['base_model']
snapshot_download(base['repo'], revision=base['revision'], token=False)
PY
```

Then select CPU explicitly before starting the backend:

```bash
CADENCE_DEVICE=cpu HF_HUB_OFFLINE=1 \
  python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The CPU profile fixes **four threads** and a **16-option limit**. The UI and results report `cpu_fp32` and explain that precision differs from CUDA. Failed CPU loading stays a CPU error; it never silently starts CUDA or another profile. Keep Python assertions enabled (do not use `python -O`) so the coordinated adapter's reconstruction checks stay active. Set offline mode before starting a fresh process; a Hub client imported before the setting is rejected. Model memory and speed depend on the host and input; the integration does not establish a serving-memory guarantee or latency SLA.

The unchanged CPU adapter uses a `public_cache` sibling of the package directory when one exists, otherwise the normal Hugging Face cache. For CUDA, an optional `CADENCE_UPSTREAM_CACHE_DIR=/path/to/released-kev-cache` passes a custom released-Kev cache to the public loader. Base/tokenizer snapshots use their normal Hugging Face cache. Device selection is bound when the service starts; restart the backend to change it. Cached runtime modules from other source directories are rejected; start a fresh process if that occurs.

### Build

```bash
npm run build
```

This produces `dist/`. Keep the backend running and use `npm run preview` to inspect the built UI at the printed loopback URL. Both Vite development and preview servers proxy `/api` to port 8000. The preview server alone does not provide model inference. The public Pages build uses `npm run build:pages` and writes `dist-pages/` at base `/cadencedemo/`; it has no inference transport. Only that directory is deployed by the Pages workflow.

## API

`GET /api/health` returns `cold`, `loading`, `ready` or `error`, plus busy state, exact revision, selected `device`, precision `profile` and limits. `cold` describes startup configuration before model loading; a healthy HTTP server is not proof that its device/cache is ready. API documentation: http://127.0.0.1:8000/api/docs.

```bash
curl http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  --data '{"state":"Can you play some jazz?","questions":{"decision":{"type":"choice","instructions":"Choose the intent.","criteria":{"music":"Play music.","weather":"Check the forecast."}}}}'
```

For Boolean use `type: "noul"` and criteria with exactly `"false"` and `"true"`. For score use `type: "score"` and an ascending array of descriptions. `state` may be nonempty text or a nonempty JSON object/array. Requests accept exactly one question named `decision`; no chat messages or generic generation parameters.

Successful responses contain `kind`, `native_option_ids`, `probabilities`, `value` (null for choice, P(true) for Boolean, expected index for score), packed `tokens`, `seconds`, `truncated: false`, model, revision, `device` and `profile` (`cuda_bf16_fp32` or `cpu_fp32`). Distributions are validated before display. IDs stay aligned with their native probability keys. Raw JSON and Copy JSON provide inspectable request/result data; editing an input marks a previous result as stale.

Demo limits: 64 KiB body, 12,000 state UTF-16 units (compact JSON length for objects/arrays), 2,000 instruction UTF-16 units, 2–32 options/levels on CUDA or 2–16 on CPU, 64-unit IDs and 500-unit descriptions. These browser-compatible counts treat BMP characters such as é or 日 as one unit and astral characters such as 😀 as two; combining marks count separately. JSON is measured without ASCII escape expansion or added separator spaces; values are preserved. The JSON editor also caps the source text at 12,000 units, including whitespace. Unpaired Unicode surrogates are rejected. Browser JSON rejects integer-valued numbers outside ±9,007,199,254,740,991 instead of silently rounding them; encode larger identifiers as strings. Other numeric JSON values use JavaScript floating-point precision. Empty choice descriptions become native null descriptions. Inputs are rejected rather than truncated. Native token limits can still reject a character-valid request. One request including model loading may run at a time; further calls receive 429. The service also limits total attempts to 12 per minute. The browser waits up to five minutes. If it times out, backend work may still be running; it retains the busy lock until completion.

This is a **loopback developer demo**, without user authentication or production job scheduling. Keep the documented loopback binding and one backend worker. Stop the backend to release the model and its device resources. GitHub Pages hosts only the static request explorer; a public inference backend and credentials are outside this change.

## Testing

Lightweight tests do not need model weights, Torch or a GPU:

```bash
python -m pip install -r backend/requirements-test.txt
python -m pytest -q
npm ci
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:ui
npm audit --audit-level=moderate
```

On hosts where the bundled Chromium installer cannot detect the OS, installed Google Chrome can be used with `PLAYWRIGHT_CHROME=1 npm run test:ui`. UI tests intercept `/api` only inside Playwright, visit `?preview=ui-test`, and mark screenshots **UI test preview / Mock result**. The query parameter only adds the label; it never supplies mocked inference in the app. CI performs validation/build/accessibility/desktop/mobile checks, never model downloads or inference.

The checked-in `.npmrc` uses npm's legacy peer resolver because npm 10 crashes resolving Vitest's optional UI/browser peers; these peers are unused. The final lockfile and clean `npm ci` are verified. This does not remove the direct Vite/Vitest compatibility requirements.

An **opt-in real inference smoke** is included. Coordinate a device slot when sharing hardware, run the real backend with model dependencies, then:

```bash
python -m backend.smoke --output docs/real-inference-smoke.json
```

This runs one choice, one JSON-state Boolean and one ordinal score request via real HTTP. It checks finite normalized distributions, revision metadata and agreement with the selected device/profile. It checks functionality, not task accuracy or performance. Native model smoke is deliberately excluded from CI. Do not run it alongside an uncoordinated shared hardware job. The checked-in real evidence covers separate reserved cached CUDA and CPU FP32 smokes, with actual environments and resource release recorded. CPU UI evidence covers all three query types without response mocking.

## Troubleshooting

| Symptom | Action |
|---|---|
| Backend not connected / non-JSON response | Start port 8000, then use the Vite development server so `/api` is proxied. |
| Model could not load | Check the backend terminal, selected device support, pinned requirements and download/cache access. CPU needs `HF_HUB_OFFLINE=1` and all pinned snapshots already cached. Retry after correcting setup. |
| Input rejected (422) | Fix malformed JSON, blank fields, duplicate IDs or limits. Shorten context/descriptions if native encoding overflows. |
| Backend busy (429) | Wait for the current load/inference to finish. Do not start a second worker. |
| Browser timeout | Inspect backend progress; the running request retains its lock. Retry after it completes. |
| Local model differs from revision | Use the exact public package; do not edit its source/config/weights. |
| CUDA out of memory | Free resources through the workload owner, shorten inputs or reduce options, then restart this backend. |

## Sources and license

The integration follows the [corrected model card and reviewed public source](https://huggingface.co/foolish-bandit/cadence-0.8b/tree/17cdf36034677e406033fbf97637b4db1984ca70), including `cadence_runtime.py` and `cadence_kev/api.py`. Model/base attribution is in the upstream package's licenses and notices. This demo repository retains its Apache-2.0 license. The unchanged coordinated CPU adapter is attributed in [NOTICE](NOTICE); its [source manifest](backend/cpu-adapter-manifest.json) is checked at load. No weights or training/evaluation datasets are committed here.
