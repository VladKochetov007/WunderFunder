# Independent review: E473 activation-mechanism screen

## Receipt and implementation checks

The fit is complete and the machine receipts agree: the ten-arm runner attempt
finished with exit 0, verified teardown, no recorded worker OOM, 25.9463
charged seconds, and 0.3838 GB peak observed GPU / 1.6403 GB peak RAM. The
attempt's only result artifact is `runs/ten_v1/result.json`; its digest and
size match the on-disk file and the scorer receipt. The fit's ten prediction
exports match all ten scorer-recorded SHA-256 digests. Both sealed specs' 13
source entries match current hashes and sizes. The fit result reports the
sealed trainer and test hashes and the frozen E470 trainer hash.

The candidate only swaps two parameter-free modules: the embedding activation
and correction-hidden activation. It keeps E470's 8,235 parameters, seeded
initialization, recurrent cell, zero correction-output initialization,
optimizer partition, data path, weighted fit objective, carried/detached state,
24 fit / 16 exposed test cohort, and 474 updates per arm. The formulas implement
the preregistered controls. The local-slope match is correct:
`GELU(x)+0.1 sin(x)` and `GELU(x)+0.1x` both have derivative 0.6 at zero;
`1.1*GELU(x)` has derivative 0.55. The tanh-GELU arm changes only the exact
versus approximate GELU formula while retaining the sine term. Since embedding
and head activations are changed together, this cannot identify which
placement drives any difference.

The sealed B4×256 probe used only fit-role sequences and completed ten
finite, parameter-changing updates. It is appropriately a compute/update gate,
not evidence for ranking arms. The CPU preflight receipt and preregistered
checks supply the implementation checks; this review did not rerun code or
score any data.

## Scoring and scientific interpretation

`score_mechanism.py` requires the successful full-fit attempt and source hashes,
then checks all ten registered arms, 474 updates, 8,235 parameters, matching
sequence order/counts, finite raw exports, identical selected targets, and
selected labels against the frozen parquet mask and clipped metric targets. It
recomputes pooled and per-target `global_wp_clip2_v1` values. It compares all
four plain-GELU and GELU+sine controls against E470, refusing a pooled-WP
deviation greater than `1e-7`. The resulting controls match exactly at the
reported precision. The grader retains the full preregistered fourteen paired
component contrasts; it does not select or prune arms based on observed scores.

The main mechanistic observation is that periodic curvature is not needed to
explain the small pilot advantage. Relative to GELU+sine, the local-linear
control is +0.001025 WP with AdamW (conditional 95% component interval crosses
zero) and +0.001809 with MuonAdamW (interval +0.000761 to +0.003468). Both
linear and sine additions beat plain GELU at this seed, while linear exceeds
sine in both point estimates. This supports treating added local slope / update
scale as a viable alternative explanation; it does not establish that the
linear variant is better beyond this panel. `1.1*GELU` is a separate scale
control, not a matched-slope comparison. Exact-versus-tanh GELU+sine differs
by only a few millionths WP; its tiny interval is not deployment evidence.

The results are conditional on one fit seed, one small one-layer GRU, 16
exposed diagnostic components, fixed nominal optimizer learning rates, and 14
exploratory contrasts without multiplicity correction. The component intervals
do not quantify training-seed uncertainty and do not support a public-score
forecast. Since AdamW and MuonAdamW retain fixed nominal learning rates rather
than optimizer-specific tuning, their differences are recipe comparisons,
not optimizer-family rankings. No dropout, schedule, activation placement, or
strong-model conclusion follows. Keep all five activations in a stronger-data
placement study; do not promote the linear arm from this result alone.

## Decision

**PASS for this bounded, preregistered mechanism screen and its scorer.** The
receipts and controls are consistent, and the report's cautious interpretation
matches the evidence. The observed result weakens the claim that periodic sine
curvature itself explains the E470 pilot gain, while preserving local slope,
global scale, optimizer interaction, and activation placement as unresolved
hypotheses. No public claim or model promotion is warranted.

Evidence: `PREREG.md`, `ROOT_REVIEW.md`, `REPORT.md`, `e473_gpu_probe_v1.sealed.json`,
`e473_ten_fit_v1.sealed.json`, `GPU_PROBE_RESULT.json`, `runs/ten_v1/result.json`,
`RESULT.json`, `score_mechanism.py`, and the two corresponding E473 runner
attempt records and logs.
