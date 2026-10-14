# Demo verification

## Browser candidate verification

The isolated candidate runs the complete trained model in Chrome 155/WASM. [Full-stack parity](browser-full-stack-parity.json) passes nine CPU FP32 fixtures, including exact 511/512-token boundaries: all leading answers agree, maximum probability drift 1.969e-6, maximum logit drift 1.717e-5 under predeclared tolerances. [Tokenizer evidence](../browser_export/tokenizer_probe_parity.json) records 115/115 exact Node/Chrome encoding cases.

[Real production UI evidence](browser-real-ui-smoke.json) covers all three native types, explicit download, verified cache reuse, corrupted-cache rejection and a token-limit error followed by successful retry without another preparation. No inference API calls, console/page errors, overflow or Axe violations. [Desktop](browser-real-ui-desktop.png) and [mobile layout](browser-real-ui-mobile.png) screenshots contain actual model results, without response mocking. Mobile hardware inference and CUDA parity are not verified.

Current checks: **81 backend tests, 47 frontend unit tests, 28 local UI cases, 12 static Pages cases and 14 browser UI cases pass**, along with TypeScript, both production builds, Ruff, whitespace checks and the online dependency audit. Browser/local automated UI tests are explicitly labeled mocks; real runtime evidence above is separate. Independent reviews cleared runtime/provenance/cache handling after fixing input-error recovery and cleared guided UX after fixing generated-ID uniqueness and readable-label collisions.

The browser artifact is published at the immutable revision in [browser-release.json](../browser-release.json). [Public verification](browser-public-artifact-verification.json) matches all 89 asset sizes/hashes and the manifest bytes. The published-build preflight checks visible opt-in download, device/size limits, guided inputs and accessibility at desktop/mobile/320px, with no automatic model/API downloads. Static fallback remains separately tested. Hosted real-inference checks are separate from this lightweight CI preflight. See [qualification, resource measurements and hosting gates](BROWSER_FEASIBILITY.md).

## Original backend release verification

Status: implementation, lightweight API integration, built UI checks and **real CUDA and CPU FP32 end-to-end smoke passed**. CPU covered all three query types through both HTTP and the built UI without API interception. No heavy CPU benchmark was run by the demo workstream.

## Evidence

| Check | Result |
|---|---|
| Python backend suite | 75 passed: three native types, JSON state, malformed/oversized/invalid inputs, load failure, context overflow, invalid distribution/timing, single-flight/load-once, shutdown, pinned-package rejection |
| Frontend contract | 37 passed, including shared multilingual/astral boundary fixtures, compact numeric JSON length, invalid Unicode, JSON numeric overflow and semantic IDs |
| TypeScript + Vite production build | Passed; ~68.6 kB local / ~67.8 kB Pages gzipped JavaScript |
| Production-build browser suite | 16 passed across desktop 1280×1000 and mobile 390×844 |
| Static Pages browser suite | 12 passed across desktop/mobile: all query types, JSON copy/validation, local setup, base assets/home navigation, zero API calls |
| Axe accessibility | Zero violations in both viewport checks |
| Browser page identity, nonblank, overlay, console | Correct title/main content, no Vite error overlay, no console/page errors in screenshot/result flow |
| Desktop/mobile overflow | No horizontal document overflow in empty and result states |
| `npm ci` | Clean install from final lockfile verified |
| `npm audit --audit-level=moderate` | Zero vulnerabilities reported |
| Ruff source check | Passed |
| Pinned source/config/learned weights | Existing supplied release package matches all nine manifest hashes; no inference run ([machine-readable evidence](model-package-verification.json)) |
| Independent code review | Four P2 findings fixed with failing-then-passing regressions: nonfinite JSON numbers, worker/model lifecycle cleanup, cached module origin verification and effective Hub offline state; Unicode follow-up counter depth regression fixed and re-reviewed with no remaining findings |

The in-app browser was attempted first and returned **“Browser is not available: iab.”** Playwright with installed Google Chrome was used as the authorized visual-test fallback. The bundled Chromium installer reported unsupported Ubuntu 26.04 on this host. CI uses Playwright's bundled Chromium on its runner.

Tests use native stubs or HTTP interception only in test files. Screenshot previews explicitly state **“UI test preview — results are mocked … No model inference”** and **“Mock result · UI test only.”** The app's preview query parameter only adds those labels; it cannot create fake outputs. The production backend has no mock mode.

## Visual comparison

The complete selected [design concept](design-concept.png) was generated using the built-in image generation tool. Both that concept and the latest browser screenshots were opened with `view_image` and directly compared. Desktop native viewport: 1280×1000; mobile: 390×844. Full-page screenshots below retain the mock labels.

Comparison points inspected: white canvas/dark typography, centered heading and supporting copy, purple segmented controls/primary action, compact example row, editor/result column ratio and borders, empty-state waveform and copy, lower query-kind guide, and mobile editor-above-result stacking. The screen follows the selected design system; no material visual mismatch remains. A full-page screenshot initially showed an offscreen skip-link artifact after focus-induced scrolling; corrected its hidden positioning and reset scroll for capture.

Above-the-fold copy matches the reference heading, supporting sentence, example labels, field labels, query/format labels and primary action. Intentional additions fulfill the working-demo requirements: mock-preview labels, input counters/hints, connection/loading/error feedback and inspectable result metadata. The score explanation corrects the generated concept's 1–5 example to the model's actual zero-based levels; Boolean guidance uses the native `noul` semantics without suggesting safety guarantees. Mobile uses 16px input text to prevent unwanted input zoom.

| State | Desktop | Mobile |
|---|---|---|
| Empty | [Mock UI preview](preview-desktop-empty.png) | [Mock UI preview](preview-mobile-empty.png) |
| Result | [Mock UI preview](preview-desktop-result.png) | [Mock UI preview](preview-mobile-result.png) |

## Interaction paths verified

Preset → edit state/instructions/options → run choice → semantic-ID distribution. Boolean preset → native `noul` request → P(true). Score preset → ascending descriptions → expected level and ordered distribution. Invalid JSON/duplicate IDs → inline error and focus. Delayed request → disabled editor and loading → completed result. API error → useful error → retry. Input edit after run → stale-result notice. Raw JSON disclosure works. Setup dialog opens and closes with Escape. Primary actions remain operable at mobile width. CPU fixture → explicit FP32 metadata → 16-option cap → probability display with precision-profile caveat is also verified.

## Real CUDA inference

The coordinator reserved a cached-only smoke slot with a hard 60-second bound. The actual demo API served one choice, one JSON-state Boolean and one ordinal score request. All returned finite normalized distributions with the pinned model revision and `cuda_bf16_fp32` device profile. The built UI then performed a fourth real choice request and rendered its native decision and precision metadata without API interception or mock labels.

[Native HTTP results](real-inference-smoke.json), [actual environment/resource release](real-smoke-environment.json), and the [real built-UI screenshot](real-cuda-ui.jpg) retain the evidence without private filesystem paths. The model process exited after **13.83 seconds**, and its GPU allocation was absent afterward. No downloads or installations occurred. Smoke evidence plus the temporary log totaled 63,757 bytes, below the 100,000-byte reservation. The log remains uncommitted.

The smoke environment used **Torch 2.11.0+cu128**, transformers 5.19.0, peft 0.21.2, safetensors 0.8.0, numpy 2.5.4, huggingface_hub 2.2.0, pydantic 2.14.0, FastAPI 0.143.0 and uvicorn 0.54.0. This is distinct from the published/benchmark Torch 2.8 environment. Observed model-call times are single functionality checks, not benchmark statistics or a latency SLA. The smoke used a temporary static mount for the same production `dist/` build and actual `create_app()` API, avoiding a separate proxy server inside the short slot.

## CPU integration boundary and limitations

The unchanged coordinated FP32 adapter is pinned by [source hash](../backend/cpu-adapter-manifest.json) and [Apache-2.0 attribution](../NOTICE). Production selection is explicit, four threads, cached-only, and limited to 16 options. CPU and CUDA startup binding, failed selection/no fallback, source/cache guards, output device/profile metadata and CPU UI limits have lightweight tests. Automated CPU metadata/limit tests use test stubs; the separate reserved native CPU smoke below verifies the real inference boundary. The separate bounded study is summarized in [benchmark and CPU guidance](BENCHMARK_AND_CPU_GUIDANCE.md); other CPU/OS portability remains unestablished. No merged or INT8 profile is recommended by this demo.


## Real CPU FP32 inference

A second coordinator reservation covered cached-only CPU FP32 inference, bounded to 60 seconds from worker launch. The integrated demo at commit `24e0b4f` served one native choice, one JSON-state Boolean and one ordinal score HTTP request. The built UI then ran all three query types with real responses, correctly displayed `cpu_fp32`, and kept its precision-difference explanation. There was no API interception, response mocking or INT8 selection.

[Native CPU HTTP results](real-cpu-inference-smoke.json), [native UI results](real-cpu-ui-smoke.json), [actual environment/resource release](real-cpu-smoke-environment.json) and the [real CPU score screenshot](real-cpu-ui.jpg) retain sanitized evidence. The worker exited after **16.66 seconds**, its process was absent afterward, and the coordinator released the held measurement queue. No downloads or installations occurred. Evidence plus the temporary log totaled **74,986 bytes**, below the 100,000-byte reservation; temporary scripts/logs remain uncommitted.

Actual profile: **CPU FP32, four threads**, Python 3.12.14 and Torch 2.11.0+cu128, with the same pinned model dependencies as the CUDA smoke. The process used the CPU even though its installed Torch wheel supports CUDA. The environment differs from the Torch 2.8 benchmark profile and is recorded without private paths. The native checks verify finite normalized distributions, query kinds, exact revision and device/profile metadata; the UI also verifies visible typed results and no uncaught page errors.

The CPU smoke establishes this demo's functionality on the tested Linux host. It does not establish task accuracy, CPU/CUDA numerical equivalence, long-context quality, other CPU/OS portability or a memory/latency SLA. FP32 remains the only exposed CPU profile; merged and INT8 alternatives are not enabled or recommended by this demo.

## Unicode limit regression

The review reproduction `{ "text": "日" × 3000 }` measured 3,011 UTF-16 units with browser compact JSON but 18,012 characters under the former backend ASCII-escaped, spaced serialization. The backend rejected it while the UI accepted it. Regression tests failed before the fix and pass with the shared browser-compatible budget. Twelve fixtures exercise accented, CJK, Arabic, astral and multilingual-key states at or below the exact boundary and one unit above it. Deeply nested bounded arrays/objects also preserve their former acceptance. Eleven shared number/escape fixtures cover compact JSON formatting. Instructions, IDs and all three kinds of descriptions also test astral boundaries; invalid surrogate inputs are rejected.

The real FastAPI validation path with a native test stub checks acceptance/rejection and that accepted state reaches the loader unchanged. Frontend contract tests independently evaluate the same fixtures. Desktop/mobile browser tests send a mixed CJK/accent/emoji state and verify the complete JSON payload and editor remain unchanged; HTTP responses are explicitly mocked for those automated tests. These are validation regressions, not additional model inference or multilingual accuracy evidence. The 64 KiB byte cap, option caps and native context guards remain independent limits.

## Import isolation follow-up

Harmless authored fixtures reproduced extra module/package dependency shadowing despite successful manifest verification. They failed before the fix. The loader now permits only the five pinned Cadence source modules, leaves `sys.path` unchanged, uses an empty model-package search path and preserves lazy native imports. It hashes the bytes it compiles, ignoring unverified cached bytecode. Tests cover clean relative/lazy imports, repeated loads, altered source before import, matching-filename foreign cache entries, extra child modules, an ancestor-path root alias and cached dependency origins. The original public package path stays unchanged for weights and the sibling CPU cache.

A lightweight check of all nine public manifest hashes imported the actual pinned `cadence_runtime` and `cadence_kev.api`, parsed a native typed request and exercised pure softmax while leaving `sys.path` unchanged. Torch was not imported and no inference ran. The retained real CUDA/CPU smokes above predate this import-isolation follow-up; no new model smoke was run during it.

## Static Pages behavior

The separate Pages build labels itself “Static playground · no model inference.” Its meaningful path is example → edit → inspect validated request → copy exact JSON → local setup. The public build has no inference transport or mock outputs. Automated desktop/mobile tests use its actual assets at `/cadencedemo/`, without HTTP interception, and assert zero `/api/` requests. The backend contract test confirms no CORS response permission and a rejected Pages-origin JSON preflight. This website does not promise direct hosted-to-local inference; real inference uses the separate local UI and same-origin Vite proxy. [Hosting details](HOSTING.md) document deployment scope and limitations.

Full-PR pre-merge review found two further P2 editor issues: repeated active-JSON selection changed state, and browser validation omitted the API byte cap. Both were reproduced with failing tests and fixed: selecting the current format preserves context, and the exact compact request permits 65,536 UTF-8 bytes but rejects 65,537. Static JSON copy uses compact serialization to preserve that byte budget. Final re-review found no remaining actionable issues. The requested Demo badge/title and expandable plain-English usage explanation are covered by both rendered build modes.

A late automated review identified unsafe-integer rounding during the initial merge command. A scoped follow-up rejects integer-valued browser JSON numbers outside the safe range, with positive/negative/nested regressions and preserved safe endpoints/string IDs. Both rendered modes show a useful error before sending or labeling a rounded request validated. The static build had already completed deployment when cancellation was attempted; follow-up publication replaces it after reviewed green CI.
