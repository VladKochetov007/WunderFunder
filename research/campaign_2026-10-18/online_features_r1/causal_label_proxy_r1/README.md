# Causal lagged-target proxy for online expert weighting

## Question

At step t, can a model use only the current observable state x[t] and its
one-step change to predict the previous-row label y[t-1]? The target horizon
and maturity are undocumented, so this is a lagged pseudo-label feasibility
screen—not evidence that a realized label is available online. Any proxy-driven
expert weighting requires a separate fixed-endpoint experiment.

## Frozen screen

The recipe fits one weighted ridge proxy on 32 F0 fit components, sampling every
10th eligible time step from complete sequences. It evaluates every eligible
step on 32 distinct F0 evaluation components selected from the source manifest
before any evaluation targets are read. The predictor uses all 112 current
features and their one-step differences; training targets are clipped to ±2 and
weighted by their clipped magnitude. The official global weighted Pearson is
applied to the shifted y[t-1] labels with the matching shifted scoring mask.

F0 evaluation components are exposed and have been used by earlier research.
The screen is discovery evidence, not clean validation. The bootstrap conditions
on one fitted proxy and does not estimate training-seed or public-transfer
uncertainty. A weak result narrows this particular pseudo-label path without
closing covariate-only online updates or the broader feature family.

The independent [implementation and claim review](IMPLEMENTATION_AND_CLAIM_REVIEW_R1.md)
found no split, alignment, mask, or metric blocker. It corrected the unsupported
realized-feedback wording and prompted stable centered weighted covariance
accumulation before any fit.

## Resource and provenance notes

The full train Parquet has a 60,437,285-byte footer. At a 1.0 GB transient cap,
PyArrow metadata construction was OOM-killed; the parent slice's local OOM
counter did not change. A corrected 1.5 GB metadata-only pass completed at
1,036,624 KiB process peak RSS. A prior matched sequence-reader replay measured
1,125,679,104 bytes of worker-cgroup peak memory. The frozen fit/grade workers
therefore reserve 1.5 GB, read one 20,000-row group at a time, and keep outputs
on physical storage under research_tmp. The failed undersized preflight is
preserved in the parent runtime record.

The exact proxy fit path was then profiled on one complete selected train group
(component 6, row group 6): the durable worker peaked at 1,067,388,928 bytes
under a 1.5 GB cap, completed in 5.03 seconds, and added no worker or parent
OOM events. This is resource evidence only; the full fit and evaluation grade
remain separate attempts. See the [profile receipt](profile_receipt_r1.json).

The frozen fit, grade and independent regrade are complete. The
[hypothesis card](hypothesis_card.json), [recipe](recipe.json), [runner](run_proxy.py),
[stable weighted-moment implementation](weighted_ridge.py), and
[execution receipt](execution_receipt_r1.json) record exact procedure and
provenance. The parent campaign README and unified research indexes link the
result and its independent review.

## Result

The frozen 32-group fit and exposed-F0 grade are now complete; the current
result and its limits are in [RESULTS_R1.md](RESULTS_R1.md). The lagged proxy
scores 0.521679 WP, with a conditional component interval [0.4488578, 0.5768614],
and an independently implemented regrade matches exactly. This establishes
previous-row-label predictability on the fixed panel only. No online gate has
been evaluated yet.
