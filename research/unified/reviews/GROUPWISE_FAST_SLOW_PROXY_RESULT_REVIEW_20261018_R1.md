# Groupwise fast/slow proxy result review R1

**Disposition: PASS for the stated negative, exposed train-row proxy result.** This review checked the frozen result and grade receipts, fit/prediction provenance, independent static grader review, and reported fold/bin arithmetic. It did not open Parquet payloads, execute the grader, or independently recompute scores.

## Findings

- The copied [grade result](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/grade_result_train_proxy_r1.json) has SHA-256 `fbd1a5dd347c763c67cc45215304a2e5148845c721172adaba0c9090a6b8fd66`; the [grade receipt](../../campaign_2026-10-18/online_features_r1/group_fast_slow_r1/grade_receipt_train_proxy_r1.json) has SHA-256 `720c5e9968748a99cf85d79e02fdfcede98b2b7e5ed702c8808c57727eed401c` and binds the same result hash.
- Pooled WP is 0.26885894 for raw, 0.26875090 for raw plus scalar clock (Δ −0.00010803; paired component-bootstrap 95% interval [−0.00243463, +0.00189382]), and 0.25576581 for raw plus 39 group states (Δ −0.01309313; interval [−0.03369300, +0.00595564]).
- The group-state point delta is negative in all four cross-fit folds (−0.000193, −0.005309, −0.036437, −0.014126) and all four position bins. The largest loss is −0.062789 WP at steps 500–1,999. The scalar clock is mixed by fold and effectively tied overall.
- The reported 636,832 scored rows equal 32 components × 19,901 post-warmup rows. Since `train.parquet` has no `is_scored`, these are all target-bearing rows after step 98. The recipe uses the official metric formula but a custom mask, so this is not an official validation score.
- The grade attempt completed in 5.02 seconds at 1,092,571,136 worker bytes under a 2 GB reservation, with zero worker/parent high, max, OOM or OOM-kill event deltas and no GPU. The label-blind fit/prediction attempt completed at 1,252,581,376 bytes, also under 2 GB with zero memory-event deltas.

## Interpretation and limits

The result supports deferring the exact 39-state additive weighted-ridge encoding. It does not establish that the states have no conditional value for a nonlinear recurrent model, and does not test student transfer, mature feedback, long-horizon adaptation or public performance. The 32 source-qualified components had already been used in feature diagnostics; cross-fitting prevents each component's labels from fitting its own prediction model but does not make the sample fresh. Bootstrap uncertainty conditions on the four fitted cross-fit models and excludes seed, sample-selection and transfer uncertainty. The data identity binds footer/metadata, not a full payload hash.

**B300 decision:** no new PR is warranted by this result. The broader causal-innovation and learned Kalman teacher/student routes already have teammate drafts; keep them open. The [existing causal-innovation draft #14](https://github.com/VladKochetov007/WunderFunder/pull/14) now includes this result as related evidence and explicitly states that it does not test that PR's projected-innovation arm. Draft #17 remains the separate learned smoother teacher/student proposal. A matched recurrent interaction is a materially different treatment and would need its own bounded local evidence before stronger claims.

## Provenance

Experiment `E-GROUP-FAST-SLOW-INCREMENTALITY-20261018-R1`; hypothesis `H-GROUP-FAST-SLOW-INNOVATIONS-20261018-R1`. The independent static [grader review](GROUPWISE_FAST_SLOW_PROXY_GRADER_REVIEW_20261018_R1.md) did not rerun the result. The copied hashes above match the durable scratch outputs.
