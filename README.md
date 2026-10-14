# Cadence demo

A local playground for [Cadence 0.8B](https://huggingface.co/foolish-bandit/cadence-0.8b): give it context, a question and criteria, then inspect a typed decision distribution. Cadence is a ~758M parameter decision model. It **does not generate chat or text**.

- **Choice** — labeled options, returned by their semantic IDs.
- **Boolean** — native `noul` query; the answer is P(true).
- **Score** — ordered level descriptions; the answer is the probability-weighted mean of **zero-based** level indices. Five levels span 0–4, not 1–5.

Probabilities are model outputs, **not calibrated guarantees of correctness**. Examples illustrate the interface; they are not a generalization benchmark. The published qualification covers training-seen task families. Long-context and unseen-task behavior are not established.

## Verification status

The local backend uses real inference only. Mocked responses exist solely in automated tests, and screenshot previews label them explicitly. Real end-to-end inference is **pending a coordinated GPU smoke test**; do not interpret passing UI/backend stub tests as model inference validation. See [verification evidence](docs/VERIFICATION.md).

## Local setup

Requirements: Node.js 22.12+, Python **3.12**, compatible NVIDIA CUDA hardware/driver, and access to Hugging Face on first use. The reviewed public loader requires CUDA and BF16; it **rejects CPU and MPS**. A separately validated CPU runtime can be integrated later; this demo does not claim CPU support.

Create an isolated environment in this repository:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r backend/requirements.txt -r backend/requirements-model.txt
npm ci
```

The model dependencies match the published CUDA profile. CUDA driver/wheel compatibility is a local requirement. The demo pins the public package at **`17cdf36034677e406033fbf97637b4db1984ca70`**, the documentation correction after release `de5009eeca68c08110e12bcd44c08016c5fe871d`; adapter/head and runtime code are unchanged. Source/config/learned-weight SHA256 hashes are checked against [the manifest](backend/model-manifest.json), including when using a local package directory. The upstream loader also verifies its pinned dependencies and tensor inventory.

Start two terminals in this repository:

```bash
# Terminal 1: activate .venv, then run ONE worker. Do not use --reload with a loaded GPU model.
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 2
npm run dev
```

Open **http://127.0.0.1:5173**. Vite proxies `/api` to the backend. Pick an example, edit its state/instructions/criteria and click **Run decision**. The first valid request downloads and loads the pinned model; allow several minutes. Later requests reuse it. Starting either server alone does not load the model or occupy the GPU.

Fresh use downloads approximately **1.8 GB** of upstream artifacts, plus the ~24 MB Cadence package. The model card reports observed CUDA allocation near 1.6 GiB on its test hardware; this is not a memory requirement guarantee. Inputs and surrounding GPU workloads affect memory and latency. Coordinate shared GPU use before starting inference.

The UI runs locally and sends input only to your backend. Model setup contacts Hugging Face. There is no analytics, request-history database or browser persistence. Errors omit input text; backend logs include diagnostic exceptions. The result's `seconds` measures the model call including encoding and CPU transfer; displayed timing excludes loading, HTTP overhead and UI rendering.

### Offline or existing local package

Prepare the pinned snapshots and upstream dependencies before going offline. For a local copy of the exact public package:

```bash
export CADENCE_MODEL_DIR=/absolute/path/to/pinned/package
export HF_HUB_OFFLINE=1
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The package directory must match every manifest hash; arbitrary or modified runtimes are rejected before import. `HF_HUB_OFFLINE=1` also stops upstream Transformers/Hub downloads. Default use downloads the pinned public package through the normal Hugging Face cache. No token is required for the public package. Upstream artifacts must already be cached for offline use.

### Build

```bash
npm run build
```

This produces `dist/`. Keep the backend running and use `npm run preview` to inspect the built UI at the printed loopback URL. Both Vite development and preview servers proxy `/api` to port 8000. The preview server alone does not provide model inference. No public deployment is included.

## API

`GET /api/health` returns `cold`, `loading`, `ready` or `error`, plus busy state, exact revision and limits. `cold` means the model has not been loaded; a healthy HTTP server is not proof of available CUDA. API documentation: http://127.0.0.1:8000/api/docs.

```bash
curl http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  --data '{"state":"Can you play some jazz?","questions":{"decision":{"type":"choice","instructions":"Choose the intent.","criteria":{"music":"Play music.","weather":"Check the forecast."}}}}'
```

For Boolean use `type: "noul"` and criteria with exactly `"false"` and `"true"`. For score use `type: "score"` and an ascending array of descriptions. `state` may be nonempty text or a nonempty JSON object/array. Requests accept exactly one question named `decision`; no chat messages or generic generation parameters.

Successful responses contain `kind`, `native_option_ids`, `probabilities`, `value` (null for choice, P(true) for Boolean, expected index for score), packed `tokens`, `seconds`, `truncated: false`, model and revision. Distributions are validated before display. IDs stay aligned with their native probability keys. Raw JSON and Copy JSON provide inspectable request/result data; editing an input marks a previous result as stale.

Demo limits: 64 KiB body, 12,000 state characters (serialized length for JSON), 2,000 instruction characters, 2–32 options/levels, 64-character IDs and 500-character descriptions. Empty choice descriptions become native null descriptions. Inputs are rejected rather than truncated. Native token limits can still reject a character-valid request. One request including model loading may run at a time; further calls receive 429. The service also limits total attempts to 12 per minute. The browser waits up to five minutes. If it times out, backend work may still be running; it retains the busy lock until completion.

This is a **loopback developer demo**, without user authentication or production job scheduling. Keep the documented loopback binding and one backend worker. Stop the backend to release the loaded model's GPU memory. Deployment, hosting and credentials are outside this change.

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

An **opt-in real inference smoke** is included. Coordinate a GPU slot, run the real backend with model dependencies, then:

```bash
python -m backend.smoke --output docs/real-inference-smoke.json
```

This runs one choice, one JSON-state Boolean and one ordinal score request via real HTTP. It checks finite normalized distributions and revision metadata. It checks functionality, not task accuracy or performance. Do not run it in GPU-free CI or alongside an uncoordinated shared GPU job.

## Troubleshooting

| Symptom | Action |
|---|---|
| Backend not connected / non-JSON response | Start port 8000, then use the Vite development server so `/api` is proxied. |
| Model could not load | Check the backend terminal, CUDA availability/driver, pinned requirements and download/cache access. Retry after correcting setup. |
| Input rejected (422) | Fix malformed JSON, blank fields, duplicate IDs or limits. Shorten context/descriptions if native encoding overflows. |
| Backend busy (429) | Wait for the current load/inference to finish. Do not start a second worker. |
| Browser timeout | Inspect backend progress; the running request retains its lock. Retry after it completes. |
| Local model differs from revision | Use the exact public package; do not edit its source/config/weights. |
| CUDA out of memory | Free resources through the workload owner, shorten inputs or reduce options, then restart this backend. |

## Sources and license

The integration follows the [corrected model card and reviewed public source](https://huggingface.co/foolish-bandit/cadence-0.8b/tree/17cdf36034677e406033fbf97637b4db1984ca70), including `cadence_runtime.py` and `cadence_kev/api.py`. Model/base attribution is in the upstream package's licenses and notices. This demo repository retains its Apache-2.0 license. No weights or training/evaluation datasets are committed here.
