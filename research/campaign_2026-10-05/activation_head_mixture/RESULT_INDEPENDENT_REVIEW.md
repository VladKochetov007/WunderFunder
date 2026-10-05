# E477 v6b export-only independent review

Reproduction passed from the sealed v6b result and paired raw export. No model was fitted and no new evaluation rows or targets were loaded; this audit uses the previously exposed 16 test components only.

- Result SHA-256: `7a6fa967845ad76fab00eb3e8285b3181fb3094d789f14ab734c093769ddf839`.
- Paired prediction export SHA-256: `1cec0f568d1898c70fb00c427f686b81afe82e065398c31db3ba804ba405a8ba`.
- Controller grade attempt: [`e477-head-mixture-cpu-test-grade-v6b.json`](../../campaign_2026-10-04/runner/attempts/e477-head-mixture-cpu-test-grade-v6b.json), SHA-256 `f3b689f9e5b63ac5e55a162d9a947436856d5e8513e3891e55372ca2cb358684` (finished, exit 0, teardown verified; receipt-bound result and NPZ hashes match).
- Independent reproducer SHA-256: `0d7294919dafdc1df3316e6d3b82d4ab7a8bd24c237ef06876d41ce45e70634b`.
- Reproduction receipt SHA-256: `12f84d2056e6b7b1b4c71f8f62673f4b9ea2d42b3e90c6f071b4fe70fbee851f`.

All 12 global and target-specific WP scores recomputed from the raw exported predictions agree with `RESULT.json` within `2e-7`. All 18 registered contrasts (14 primary, 4 secondary) were recomputed with 2,000 paired component bootstrap draws per fit seed using the same numeric-sorted component draw mapping. Source arrays passed the raw-finiteness and alignment checks before metric clipping.

The small exposed-component comparison is mixed across seeds: channel-split minus GELU at dropout 0 is -0.01740 for seed 271828 and +0.00698 for seed 271829; same-channel mixture minus GELU is -0.00575 and +0.00526, respectively. The preregistered mixture-by-dropout interaction is +0.00615 and +0.00075 in the two seeds. These are conditional diagnostics on 16 exposed components from two fit seeds, with 18 registered contrasts and no multiplicity adjustment. They do not select a winner or support public-score inference.

The independent receipt is `eda/twelve_v6b/RESULT_INDEPENDENT_REVIEW_V6B.json`. The earlier v6 attempt failed only after writing the export because a relative output path could not be recorded relative to the repository root; that attempt and its artifacts were retained. The successful v6b grade used the same frozen scorer and an absolute output path; no fits were repeated.
