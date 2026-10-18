# Groupwise fast/slow Kalman innovation states

**Status: the exact one-sequence profile, label-blind four-fold component cross-fit, and separately labeled train-row proxy grade completed. The additive group-state ridge arm lost on the exposed proxy; this narrows that representation only.**

This follow-up asks whether the all-channel innovation signal retains useful predictive information when represented as 39 causal group states: fast and slow EWMA innovation energies and their log ratio for each of the 13 input groups. It compares the 151-column raw-plus-group representation with the same 112-column raw baseline and a 113-column raw-plus-scalar-clock reference.

## Frozen experiment

- [Recipe](recipe.json) pins the fit-only source groups, component cross-fit, feature equations, target alignment, weighted ridge configuration, official metric and resource bounds.
- [Hypothesis note](../../../unified/ideas/GROUPWISE_FAST_SLOW_KALMAN_INNOVATION_INCREMENTALITY_20261018_R1.md) explains the mechanism and the evidence required for a recurrent follow-up or B300 handoff.
- [Feature extractor](group_features.py) keeps raw inputs and computes group states from pre-update, clipped Kalman innovations. State resets per sequence and does not read labels or scoring masks.
- [Cross-fit runner](run_cv.py) fits each fold on 24 components and produces target/mask-blind predictions for its eight held-out components. [Stable ridge accumulation](ridge_model.py) reuses the previously reviewed centered weighted-moment implementation.
- [Official-mask grader](grade_cv.py) reads labels and masks only after prediction arrays are frozen and hashed. It cannot produce an official score from `train.parquet`, which has no `is_scored` column.
- [Train-row proxy grader](grade_train_proxy.py) uses all prediction-eligible train rows (steps 99–19,999) after OOF predictions are frozen. Its separate [grade addendum](grade_train_proxy_r1.json) binds this mask policy and explicitly labels the output as a nonofficial development proxy; the frozen fit recipe remains unchanged.
- [One-sequence profile](profile_one.py) checks the exact feature and fit-side ridge path under an admitted no-swap memory cap before the full cross-fit run.

The 32 components were selected for an earlier feature-only diagnostic and are repeatedly exposed development data. Internal cross-fitting prevents a component's labels from fitting its own prediction model, but the sample is not a fresh validation set. Any positive result is a small-data incrementality screen, not an endpoint, student, competition or public-transfer claim.

## Current execution status

- Source and recipe files pass Ruff formatting and lint checks.
- An independent static review passed after correcting three import/recipe blockers. See the [review note](../../../unified/reviews/GROUPWISE_FAST_SLOW_SCREEN_RECIPE_REVIEW_20261018_R1.md).
- The independent proxy-grader review passes the mask interpretation and source/provenance checks; it reviewed code only. See the [review note](../../../unified/reviews/GROUPWISE_FAST_SLOW_PROXY_GRADER_REVIEW_20261018_R1.md).
- The exact one-sequence profile completed in 5.02 seconds at 1,122,492,416 bytes worker peak under a 2 GB cap, with zero worker or parent memory-event deltas and no GPU use. See the [operations note](../../../unified/operations/GROUPWISE_FAST_SLOW_FEATURE_PROFILE_20261018_R1.md) and [profile receipt](profile_receipt_r1.json).
- The full label-blind fit/prediction completed in 35.16 seconds at 1,252,581,376 bytes worker peak under a 2 GB cap; worker and parent OOM/max event deltas were zero. Frozen OOF predictions and models remain in physical scratch, bound by [fit/prediction receipt](fit_predict_receipt_r1.json).
- The first grader exited after 5.02 seconds at 1,048,162,304 bytes peak, with zero OOM/max deltas. It stopped because the training source has no explicit scoring mask, not because of resource pressure. The mask contract is in [`docs/data_overview.md`](../../../../../docs/data_overview.md).
- Because `train.parquet` does not contain `is_scored`, the separate proxy grader scored every target row at steps 99–19,999. Raw scored 0.268859 WP, raw plus the scalar clock 0.268751 (Δ −0.000108; paired component-bootstrap 95% interval [−0.002435, +0.001894]), and raw plus 39 group states 0.255766 (Δ −0.013093; interval [−0.033693, +0.005956]). The group-state arm lost in all four component folds and all four position bins, with its largest bin loss at steps 500–1,999 (−0.062789 WP). These are proxy scores using the official metric formula but a different mask, not official validation results. See the frozen [proxy result](grade_result_train_proxy_r1.json), [grade receipt](grade_receipt_train_proxy_r1.json), and independent [result review](../../../unified/reviews/GROUPWISE_FAST_SLOW_PROXY_RESULT_REVIEW_20261018_R1.md).
- The independent result review matched the result and resource hashes and confirmed the fold/bin arithmetic. It did not reread Parquet payloads or independently recompute scores. The sample has 32 components already exposed by prior feature work, the bootstrap conditions on four fixed fits, and source identity pins the footer/metadata rather than a full payload hash.
- Decision: defer the exact 39-state additive ridge encoding. This does not reject learned recurrent use of Kalman states or the broader causal-innovation/teacher-student direction. Existing teammate handoffs [#14](https://github.com/VladKochetov007/WunderFunder/pull/14) and [#17](https://github.com/VladKochetov007/WunderFunder/pull/17) already cover the broader families; the #14 draft description was updated with this proxy result as related evidence, not as a test of its projected innovation arm. This proxy does not justify a separate B300 PR. A recurrent interaction screen would need a matched raw-versus-state comparison on complete sequences before any stronger claim.

All generated outputs belong under physical scratch at `/home/vlad/development/wunder/research_tmp/online_features_r1/group_fast_slow_r1/`; no experiment artifacts are written to RAM-backed temporary storage.
