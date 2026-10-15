# A shorter path to a real decision

The demo starts with a legal-work example and a readable preview of its actual current context, question and answer criteria. Desktop shows six task cards; phones use a native selector. Full editing is optional, and its controls mount only when opened. Editing updates the preview and preserves native IDs. The stage action is Download model / Use saved model, then Get decision. On phones the action remains reachable at the bottom, with download requirements beside consent and keyboard focus kept above the bar.

The first screen identifies a research demo, local execution and the possibility of wrong outputs. The 1.54 GB model plus 28 MB runtime remain explicit. No model worker, manifest, weights or runtime download starts merely by opening the page or choosing a task. A lightweight local cache-key check can say saved files exist; it never calls them verified. Preparation still verifies every saved file's size and SHA-256 and downloads missing/invalid files only after a click.

Results lead with the readable outcome, then the full distribution and correctness caveat. Rating rows pair level numbers with their supplied labels. Raw JSON, model revision, precision, storage controls and explanatory material remain available through disclosures. All six fictional legal-work defaults and native choice/Boolean/ordinal semantics are unchanged. This remains a general-purpose research model, not validated legal advice or document generation.

## Safe loading changes

- Download directly into one buffer sized by the pinned manifest, rather than keeping chunk buffers and copying them into a second full-file assembly. Short, oversized and failed streams are rejected; expected bytes alone do not establish EOF.
- Remove explicit full-file copies before hashing and cache writes. Hashes respect the exact input view, including nonzero offsets. Browser APIs may still make internal copies; no peak-memory measurement is claimed.
- Rate-limit streamed-byte UI updates to roughly 100 ms while always reporting file/phase/final-byte transitions. The transfer counter still uses received bytes, stays below 100% before EOF and distinguishes verification from readiness.
- Stop rebuilding preparation snapshots for inference status. Show actual completed work steps separately from elapsed time, without treating them as an ETA.
- Use aligned native 16-bit views for lossless BF16 expansion on little-endian hosts, with the original DataView fallback for other views. An exhaustive test checks all 65,536 bit patterns on aligned and unaligned input. No numeric rounding, precision, graph/provider, model file, artifact pin or thread-count change is made.
- Retain size/SHA verification immediately before every cached file is used. Do not retain all layer sessions or decoded weights; layers still release sequentially to limit memory demand.

The qualified 28 MB WASM binary is unchanged. Switching inference binaries or quantizing the model would require separate parity and device qualification. Emitting another runtime asset during a build does not mean the browser downloads it; real network checks distinguish these.

## Measured baseline and verification scope

Baseline: deployed commit `0cb0d7c7da2a512b870e5af306588294f45c441d`, Chrome 155 on Linux, 390×844 viewport, existing verified model cache, unthrottled network/CPU. Three fresh-page first-decision samples used the same 86-token document-routing request and 89-file cache. They recorded:

| Measure | Before |
| --- | ---: |
| Initial page height | 3,029 px |
| Context starts | 1,116 px below page top |
| Cached preparation, median | 4.94 s |
| First decision wall time, median | 10.05 s |
| Initial JS/CSS/resource decoded bytes | 248,985 B |

Paint samples were 280/56/68 ms with warmed HTTP assets after the first navigation. These are observations of this browser/profile/network, not cold mobile startup estimates. A real remaining tokenizer-config download measured 16,712 bytes; its transfer-to-verification interval was 132 ms in that run. The full 1.54 GB download was not repeated for this audit, so no new full-download duration or network-speed claim is made.

The first local redesign measurement reduced page height to 1,610 px and context start to 534 px, with median cached preparation 3.94 s. This preceded the final compact phone selector. Its median decision time was 10.58 s while UI suites were also running, so it does not establish an inference speedup. Final live measurements and deployment evidence are recorded separately after publishing; local asset delivery is not comparable to the public CDN.

Regression coverage includes complete/short/oversized/interrupted streams, EOF before completion, exact-view hashing, exhaustive BF16 bit identity, accurate download progress, cache discovery without automatic download or verification claims, stage transitions, cancellation/retry, native request semantics, optional editing and keyboard visibility above the action bar. Mocked UI tests remain labeled; real browser model checks are separate and compare observed distributions to prior results.

## Remaining large-model limits

A 1.54 GB download and a 758M-parameter model remain substantial for a phone. Desktop is recommended. A phone may lack memory/storage, evict cached files or stop a background tab; preparation or inference can fail. Keep the tab open, use a suitable connection and retry only when ready. Actual iOS hardware/Safari has not been tested. Browser measurements do not validate legal accuracy, calibration, broad generalization or mobile performance. The 512-token/16-answer browser limits, explicit errors and local-backend setup path remain in place.
