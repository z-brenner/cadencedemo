# Demo verification

Status: implementation, lightweight API integration and built UI verification complete. **Real CUDA end-to-end smoke pending coordination.** No GPU inference or heavy CPU benchmark was started during this phase.

## Evidence

| Check | Result |
|---|---|
| Python backend suite | 12 passed: three native types, JSON state, malformed/oversized/invalid inputs, load failure, context overflow, invalid distribution/timing, single-flight/load-once, shutdown, pinned-package rejection |
| Frontend contract | 4 passed, including JSON numeric overflow regression and semantic IDs |
| TypeScript + Vite production build | Passed; ~67.5 kB gzipped JavaScript |
| Production-build browser suite | 8 passed across desktop 1280×1000 and mobile 390×844 |
| Axe accessibility | Zero violations in both viewport checks |
| Browser page identity, nonblank, overlay, console | Correct title/main content, no Vite error overlay, no console/page errors in screenshot/result flow |
| Desktop/mobile overflow | No horizontal document overflow in empty and result states |
| `npm ci` | Clean install from final lockfile verified |
| `npm audit --audit-level=moderate` | Zero vulnerabilities reported |
| Ruff source check | Passed |
| Pinned source/config/learned weights | Existing supplied release package matches all nine manifest hashes; no inference run ([machine-readable evidence](model-package-verification.json)) |
| Independent code review | Two P2 findings fixed with failing-then-passing regressions: nonfinite JSON numbers and worker/model lifecycle cleanup |

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

Preset → edit state/instructions/options → run choice → semantic-ID distribution. Boolean preset → native `noul` request → P(true). Score preset → ascending descriptions → expected level and ordered distribution. Invalid JSON/duplicate IDs → inline error and focus. Delayed request → disabled editor and loading → completed result. API error → useful error → retry. Input edit after run → stale-result notice. Raw JSON disclosure works. Setup dialog opens and closes with Escape. Primary actions remain operable at mobile width.

## Remaining check

After the GPU coordinator supplies a safe slot, run the **real** backend with the pinned model dependencies and the supplied verified local package/cache. Run `python -m backend.smoke` for one choice, JSON Boolean and score HTTP request, then inspect a real UI result and replace this pending status with the exact evidence. This is a functionality check, not an accuracy claim or benchmark. No end-to-end model success is claimed yet.
