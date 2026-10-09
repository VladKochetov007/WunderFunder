# E-518: exact-metric calibration of a weak-model disagreement gate

**Status:** completed exposed-data screen; independent post-score audit passed.  
**Question:** Did E-517's zero gate arise only because it fitted coefficients
with weighted squared error rather than the official clipped weighted-Pearson
metric?

## Frozen comparison

E-518 reused E-514's fixed `ab12` predictions, complete-chain three-fold split,
E-517's target-0 row-disagreement gate and matched global blend. It changed only
coefficient selection: each calibration fold searched the registered 101-point
beta grid using exact target-0 `global_wp_clip2_v1`; target 1 stayed at the
anchor. Evaluation used once-held-out groups and the official pooled two-target
endpoint. All 123 chains / 432 groups were already exposed, so this is a method
screen only. The recipe and pass rule are frozen in
[`METRIC_ALIGNED_GATE_PREREGISTRATION_E518.md`](METRIC_ALIGNED_GATE_PREREGISTRATION_E518.md).

## Result

| Endpoint | Official pooled WP | Delta vs anchor |
|---|---:|---:|
| C1 anchor | 0.6832264635 | — |
| Gated `ab12` | 0.6832229115 | −0.0000035520 |
| Global `ab12` control | 0.6832256788 | −0.0000007847 |

The gated endpoint trails the anchor by `3.55e-6` and the global control by
`2.77e-6`. Paired complete-chain 95% intervals are `[-0.00010082,
0.00007918]` versus the anchor and `[-0.00008352, 0.00007074]` versus the
global control. Gate betas by fold are `0.00`, `0.01`, and `0.02`; only the
positive-beta-count condition passed. The registered endpoint and interval
conditions failed, so `development_gate_passed` is false. The exact-metric
objective correction does not rescue this fixed pool/gate recipe.

The fold-level and pooled deltas with paired intervals are easier to inspect in
[`plots/E518_DELTA_DETAIL.png`](plots/E518_DELTA_DETAIL.png). The original
endpoint and coefficient chart is in
[`plots/E518_METRIC_ALIGNED_GATE.png`](plots/E518_METRIC_ALIGNED_GATE.png).

## Execution and evidence limits

The independent pre-score reviewer passed the corrected global-maximum beta
tie rule and the zero-weight return path; focused tests passed 4/4. The Marimo
kernel's UI setter did not schedule the saved button-dependent action, so the
registered runner was invoked once from that same Marimo kernel's code-mode
scratchpad. The attempt marker, unchanged reviewed runner, frozen input hashes,
OOF prediction digests and plot digest are preserved in
[`E518_PRE_SCORE_ADMISSION_20261007.json`](E518_PRE_SCORE_ADMISSION_20261007.json),
[`E518_ATTEMPT_STARTED.json`](E518_ATTEMPT_STARTED.json),
[`E518_RESULT.json`](E518_RESULT.json), and
[`E518_POST_RUN_EXECUTION_RECEIPT_20261007.json`](E518_POST_RUN_EXECUTION_RECEIPT_20261007.json).
The independent reviewer recomputed the saved OOF arrays, endpoints and
bootstrap; all stored hashes and values matched. See
[`E518_INDEPENDENT_EVIDENCE_REVIEW_20261007.md`](E518_INDEPENDENT_EVIDENCE_REVIEW_20261007.md).

The screen completed in 13.2 seconds under one CPU core, 2.5 GB maximum memory,
and no swap. Scope peak memory was 1.53 GB; scope OOM counters stayed zero. No
GPU, neural fit, public submission or new public-score observation was involved.

## Bandit decision

Close E-518's exact metric-aligned gate and all further tuning on the E-514
pool, regardless of independent audit outcome. Keep the broad possibility that
a weak standalone model contributes useful information available only for a
materially new, ancestry-audited candidate cohort. There, rank members by
cross-fitted conditional residual value and the complete served endpoint,
alongside shuffled-member and anchor-only controls; raw correlation and
standalone score are diagnostics, not selection objectives.

This direction remains lower priority than recovering our independently
trained baseline/raw-teacher inputs and testing heterogeneous-teacher
compression. The current screen grants no numerical prior and no public-gain
forecast. Our own verified public best remains 0.6660; no new submission is
ready.
