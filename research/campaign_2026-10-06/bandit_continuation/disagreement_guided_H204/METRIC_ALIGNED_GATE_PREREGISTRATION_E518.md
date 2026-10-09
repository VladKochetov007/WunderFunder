# E-518 preregistration: exact-metric fitting for the fixed disagreement gate

Registered before scoring, 2026-10-07. This is one narrow, user-directed
follow-up to E-517 after its independent review. It asks whether E-517's
weighted-squared-error coefficient fit rejected the weak model even though the
competition metric is clipped weighted Pearson. It does **not** reopen general
search over the E-514 pool.

## Prior evidence and question

E-514 selected no positive global blend for its fixed members on its
previously exposed groups. It did use the exact official score on a coarse
`lambda = 0, .05, ..., .25` grid, and every selected lambda was zero. E-516
also kept anchor weights and produced a small negative point change. E-517's
independently audited rowwise-gate screen used a weighted-squared-error
surrogate to fit beta; all gate and global betas were zero, and all held-out
endpoints exactly equaled the anchor. These outcomes close their exact recipes.

E-518 tests one specific unresolved distinction: can exact
`global_wp_clip2_v1` calibration select a useful coefficient for the same
fixed rowwise disagreement gate when the WLS surrogate did not? A positive
screen result would be a local, exposed-pool lead only. Every source group has
already been inspected; no public-transfer inference or independent
confirmation is available.

## Frozen inputs and intervention

- Reuse only the E-514 V3 source manifest and chain partition, with SHA-256
  `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd` and
  `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- Candidate: `ab12`, the E-514 member with the smallest descriptive weighted
  prediction correlation to the C1 anchor. Only target 0 is modified; target 1
  remains at the C1 anchor.
- Preserve E-517's whole-chain three-fold partition, calibration/evaluation
  membership, gate feature `abs(clip(ab12)-clip(anchor))`, linear-interpolated
  75th-percentile threshold, ties-in routing, official mask/warmup, ±2
  clipping and anchor/global controls.
- Change only coefficient fitting. For the gate and global controls
  independently, select `beta` from the fixed 101-point inclusive grid
  `[0, 0.01, ..., 1]` by maximizing the **exact official clipped target-0 WP**
  on calibration chains. Target 1 is unchanged, so maximizing target-0 WP is
  equivalent to maximizing the official two-target mean for these candidates.
  If values are within `1e-12` of the grid maximum, choose the smallest beta.
  Do not select on outer-fold results.
- Score the concatenated once-held-out endpoints with `global_wp_clip2_v1`.
  Report per-fold thresholds, betas, exact calibration scores, evaluation gate
  row/official-weight mass, pooled and fold endpoints, and paired complete-chain
  bootstrap intervals stratified within the fixed outer folds.
- Use 2,000 bootstrap replicates, seed `998465967`, derived from the first
  eight hex digits of the preserved `sensors` stdout SHA-256
  `3b8361afda9622252fe8da30596f2734c8be8f1753708b3274e0c5d74cf89b17` in
  `E518_SENSORS_STDOUT_20261007.txt`.
- The registered screen passes only if the pooled gated endpoint beats both
  anchor and fitted global control, both paired 95% lower bounds are positive,
  and the gated beta is positive in at least two folds. Any failure closes
  this exact objective-corrected gate on this pool. No further same-pool
  variants are authorized by this registration.

## Implementation, resources and review

The runner is `metric_aligned_gate.py`; the paired Marimo notebook is
`E518_metric_aligned_gate_eda.py`. It reuses the audited E-514 loader and the
registered complete-chain bootstrap implementation. The E-517 pool/partition
checks are repeated before fitting. One attempt marker prevents accidental
re-execution; complete results and prediction arrays are reloaded for display.

The screen fits no neural model and uses no GPU. Admit it under one CPU core,
2,500,000,000 bytes maximum process memory, no swap, 360 seconds maximum
runtime. It must start before the user's 10:00 Europe/Kyiv local-compute cutoff;
after that cutoff any new experiment compute is Kaggle-only. The run requires
pre-score independent code review and a post-score independent evidence audit.
