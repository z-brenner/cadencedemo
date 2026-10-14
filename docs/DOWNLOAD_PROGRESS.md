# Browser download progress

The previous live UI rounded percentages to whole numbers, so an unfinished transfer at 99.5% or above could show 100%. A controlled slow-network check held the final real tokenizer configuration request: the page showed 100% with 16,712 bytes still missing. No response bytes or weights were mocked or changed. The previous cumulative counter also added verified cached files and presented them as downloaded bytes.

Preparation now has separate phases:

1. Fetch and verify the pinned manifest, then check every saved file's size and SHA-256. This phase shows a file-check count and an indeterminate bar, not a download percentage.
2. Download only missing/invalid files. The stable denominator is their combined manifest byte size. The numerator sums each file's absolute received-byte count, including its partial chunks once. Verified saved bytes are reported separately and excluded from the download total.
3. After each stream ends at its exact expected length, verify its hash and save it. This phase is labeled verification/saving, with an indeterminate bar.
4. After all files are verified and saved, prepare the tokenizer. The ready state then identifies 100% of model files as verified. Browser runtime initialization on the first decision is separate.

Displayed transfer percentages use a tenth of a percent, rounded down. They stay below 100% until all missing streams have actually ended; merely receiving the expected final byte does not declare EOF or successful verification. Unequal file sizes are weighted by bytes, not file counts. Per-file absolute counters prevent duplicate events from double counting.

Cancel terminates the worker and discards the unfinished file. Retry begins a new remaining-download calculation from reverified saved files; already verified files are reused and partial bytes from the canceled attempt are excluded. If every file is saved and valid, preparation performs no model-file download and shows cache verification followed by readiness. The pinned manifest is still fetched for each preparation.

Regression coverage includes unequal sizes, partial/repeated chunk counters, partial cache reuse, all-cached preparations, cancellation/retry, overrun/short-stream rejection, final EOF before 100%, and verification before readiness. Rendered browser tests cover rounding, cache labels and the distinct verification/loading phases. Automated response/worker fixtures remain explicitly marked UI tests; real deployed download checks are recorded separately.

Reproduction uses Chrome with a mobile viewport and a delayed real final request. This establishes the application bug and its arithmetic independently of a device-specific claim. Physical iOS testing has not been performed.
