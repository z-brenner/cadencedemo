# Guided demo interface

The demo keeps its existing native request contract and model revision. The visible form now follows Context → Question → Possible answers, with familiar answer formats: Choose an answer (`choice`), Yes or no (`noul`), and Rate on a scale (`score`).

## Interface behavior

- Start with an editable example. The examples classify requests or rate provided content; they do not perform actions or retrieve live weather.
- Context and Question use ordinary prose styling. Answer descriptions wrap on mobile. Numbered fields and a single primary form action explain the sequence.
- Advanced input details contains state format and exact native choice IDs. Display names may replace underscores and capitalize the first character; this does not modify any request value. If readable names collide (ignoring case), the editor, winning-answer summary, and probability rows append the exact native ID. Results derive labels from the saved run, including after the current inputs change. Editing an answer description preserves its existing native ID. Custom IDs remain editable in advanced details. Added answers receive the first unused `option_N` ID; removing or adding another answer never renumbers the remaining IDs.
- Local mode uses Get decision and the existing real backend. Static Pages uses Inspect request, displays No inference has run, and produces only a validated request. Request JSON is a disclosure; Copy JSON remains available.
- Choice results display Cadence favors and a readable answer name. An exact tie displays No single leading answer. All probabilities remain visible as text and bars.
- Boolean results display Model probability of Yes, using `p(true)`. Rating results display Average model rating, retaining the weighted native level index and denominator.
- Results explicitly explain that output percentages do not measure correctness. Inputs changed after a run retain the existing stale-result warning.
- Timing, packed tokens, precision, pinned revision, and result JSON copy are inside Technical result details. Raw JSON remains a separate closed disclosure.
- Existing input limits, JSON numeric safeguards, response validation, loading/error states, and setup instructions remain active.

## Browser preparation

When both `VITE_CADENCE_ASSET_BASE` and `VITE_CADENCE_MANIFEST_SHA256` are configured, the interface prepares and runs the browser worker instead of the local API. With either value missing, the existing local backend and honest static Pages behaviors remain available.

- Opening the page and editing examples create no worker and download no model weights or WASM runtime. Download model is the explicit preparation action.
- The preparation panel explains the approximately 1.54 GB model download plus 28 MB browser runtime, local input processing, and the download host seeing the device IP address.
- The panel distinguishes checking saved files, downloading, loading, ready, running, canceled, and error states. Byte counts and percentages come from worker reports. A failed or unsupported browser operation shows an error and optional local setup instructions; it never silently invokes an API.
- Cancel terminates the worker and rejects an in-flight decision. Retry can reuse completed verified files; unfinished files download again. Late events from retired workers are ignored.
- Clear saved model deletes only `cadence-browser-{configured manifest SHA}` without creating a worker or accessing the network. Cache storage is required by this runtime; preparation fails explicitly when it is unavailable.
- Browser requests use at most 16 answers and 512 packed tokens. Oversized inputs are rejected, without truncation. Technical runtime stages, cache management, limits, and revision sit inside Model and saved files.
- Browser results are validated against the exact configured artifact manifest SHA in addition to the trained model and revision. Local backend health accepts only CPU or CUDA profiles.

Browser UI tests use a mocked Worker only in the explicitly labeled `?preview=ui-test` flow. They verify UI state and request handling, not physical model execution or numerical parity; real model verification belongs to the runtime evidence.

## Verification

The repository's Vitest contract tests, both production builds, and Playwright desktop/mobile suites cover native semantics, clipboard export, JSON and Unicode validation, CPU limits, loading/recovery, stale results, hidden controls, probability ties, keyboard dialogs, accessibility, and static Pages making no inference calls. UI inference results in Playwright are explicitly labeled mocks. These tests do not establish model execution or numerical parity in a browser.

Screenshot evidence is saved under `/tmp/cadence-guided-*`; the test suite no longer rewrites tracked preview images. The desktop viewport is 1280×1000 and the mobile viewport is 390×844; a dedicated accessibility and overflow case also checks a 320px-wide phone.


`npm run test:browser` reproducibly builds the isolated fixture in `/tmp/cadence-browser-ui-dist` and runs `playwright.browser.config.ts` on port 5176. Its asset URL and manifest SHA are fixtures for the explicitly mocked Worker. The default/static suites use builds without browser asset configuration. CI runs all three suites.

Input encoding errors, including exceeding 512 tokens, settle only that request and restore readiness. The user can shorten the input and retry with the same prepared worker; corrupted-cache and runtime failures remain fatal. Fourteen browser UI cases verify this recovery, cancellation, cache isolation, model provenance and lack of API fallback. Actual model/download/cache evidence is recorded separately in [the production UI smoke](browser-real-ui-smoke.json).
