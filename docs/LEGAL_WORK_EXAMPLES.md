# Fictional legal-work examples

All six editable presets and the defaults for choice, Boolean (`noul`) and ordinal score use fictional legal-work content. The model, query contract, weights and artifact pins are unchanged. Cadence compares supplied answer criteria; it does not generate documents, retrieve citations, determine enforceability or provide legal advice. The examples illustrate an interface, not validated legal performance. Review outputs against the supplied words and use fictional inputs rather than real client information.

| Preset | Query | Task based on supplied content |
| --- | --- | --- |
| Route a document | Choice, text | Classify a confidentiality excerpt among agreement and meeting-note categories. |
| Route a review request | Choice, JSON | Route a fictional draft-agreement request to a work queue. |
| Check a notice term | Boolean | Check whether the clause explicitly states at least 30 days of written notice. |
| Check automatic renewal | Boolean | Check whether a clause requiring a new signed extension explicitly says renewal is automatic. |
| Prioritize review | Score | Rate pending review with an internal target tomorrow, using the supplied 0–4 workflow levels. |
| Prioritize an archive item | Score | Rate a completed, archived review with no remaining work using the same levels. |

Notice/renewal checks concern only the supplied wording. Priority examples use fictional internal targets, not statutory, court or other legal deadlines. No real parties, client facts, citations or jurisdiction-specific conclusions are supplied.

Mode defaults use the first example of the selected type and remain editable. JSON state is preserved as structured data. The score is a probability-weighted mean of zero-based ordered levels, not a risk estimate or a legal conclusion. Boolean output reports the model's `p(true)`, not a verified fact. Choice percentages show preference among the supplied criteria, not calibrated correctness.

UI tests exercise each preset and preserve native request semantics, default-mode switching, readable answer labels, stale-result behavior, mobile layout and accessibility. Mocked UI responses are labeled as tests. Real browser inference evidence is recorded separately with the actual distributions and pinned model provenance; observed outputs must not be represented as accuracy validation.

Older tracked smoke/parity evidence contains the scenarios used for those historical runs. Those records are preserved unchanged to avoid rewriting measured results; they are not current demo presets. The opt-in backend smoke script now uses legal-work defaults for its next run.

## Recorded real browser check

[The local-preview smoke record](legal-work-browser-smoke.json) contains the exact requests and full observed distributions for all six presets, using the real worker, unchanged pinned model and verified saved files. No inference or worker-status messages were mocked. Chrome 155 on Linux/WASM returned:

| Preset | Observed displayed output |
| --- | --- |
| Route a document | Confidentiality agreement |
| Route a review request | Contract review |
| Check a notice term | 78.8% model probability of Yes |
| Check automatic renewal | 0.9% model probability of Yes |
| Prioritize review | Average model rating 3.61 / 4 |
| Prioritize an archive item | Average model rating 0.00 / 4 (unrounded mean 0.00191) |

The pending-review example did not put all probability on the highest level: about 65.1% was on level 4 and 32.8% on level 3. All distributions are preserved without changing examples to make their outputs appear certain. Packed requests used 83–150 tokens within the 512-token browser limit. These are six bounded functionality observations, not legal accuracy validation, calibration or a performance benchmark. No physical iOS test was performed. Accessibility checks found no violations, and widths 320, 390 and 1280 had no horizontal overflow. Live verification after authorized deployment is recorded separately.
