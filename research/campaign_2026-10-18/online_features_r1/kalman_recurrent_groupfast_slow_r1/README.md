# Recurrent interaction with groupwise Kalman states

**Status: completed two-seed local screen on the same exposed F0 panel. The active-state delta changes sign across seeds and remains tiny relative to uncertainty; keep the mechanism open without promoting a stable gain.**

## Question

Does a GRU learn useful interactions with causal groupwise fast/slow Kalman innovation states when the same 39 appended columns are present but zeroed in the matched control? The prior four-fold ridge screen was negative for this encoding, but it did not test recurrent nonlinear interactions. This is the narrow next screen recommended by the independent research strategist.

## Frozen feature definition

- Preserve all112 raw input fields and existing event features.
- Normalize each sequence's input channels using only its first99 rows (mean and sample standard deviation, floor1e-6).
- Starting at row99, compute pre-update standardized innovations from a diagonal random-walk Kalman filter with `Q=0.001`, `R=1`, initial level0 and covariance1; clip innovations to ±8.
- For each of13 documented semantic groups, compute mean squared innovation, then fast and slow exponential states with spans20 and200. Append `log1p(fast)`, `log1p(slow)`, and `log(fast+1e-8)-log(slow+1e-8)`, for39 features total. Rows0–98 receive zero features.
- Every state resets at the sequence boundary. The extractor reads no labels, mask, sequence identifier or future row.

The local implementation is in [`src/wnn/kalman_group_features.py`](../../../../src/wnn/kalman_group_features.py), with trainer/checkpoint integration in the other pinned source files listed by the recipe. It supports batched full-sequence training and a one-row streaming state. Full-precision feature computation follows the frozen NumPy reference; output columns are float32.

## Preflight evidence

The [preflight script](preflight_features.py) reads only the112 input columns for one fitting-side sequence. It compares the new feature path with the earlier frozen `group_features.py` implementation; checks NumPy, batched Torch, and one-row Torch/NumPy streaming agreement; and checks a future perturbation and reset. The [receipt](preflight_result_r1.json) records the source group, feature hash and numeric errors. It does not read labels or masks and makes no quality claim.

The actual B8, 20k-row CUDA profile is recorded in [profile_receipt_r1.json](profile_receipt_r1.json). It completed60 timed loader batches (610 optimizer attempts across the chunked sequence batches), with finite losses and gradients. Peak worker RAM was4.05GB under the4.5GB cap; peak observed VRAM was1.38GB. Worker and parent OOM/max counters stayed at zero. The controller logged memory-high events near the worker threshold, so the fit pair keeps the same4.5GB hard cap and is admitted one arm at a time. The first launch request was rejected by the controller's CPU accounting before any GPU work; the successful profile used a half-core controller scope and one-core worker reservation.

## Matched comparison

The [recipe](recipe_r1.json) pairs an active-state fit against an equal-width all-zero feature control. Both start from `warm_mechanism_f0_s3_c_r3@ep1`, use the same32 complete source-qualified fit components, seed3, batch8, chunks2000, full FP32, one epoch, weighted MSE, AdamW/OneCycle and a fixed epoch-0 checkpoint. Parent-inherited auxiliary weights are also identical (`aux_next=0.3`, `aux_struct=0.3`). The active representation differs only in the values of the 39 appended features. All old parent columns remain present in both arms.

Both fits and feature-only evaluation exports completed. Active scored0.67986413 WP versus0.67981250 for zero control (Δ+0.00005163). A paired141-component bootstrap gave a conditional 95% interval [−0.00013143,+0.00027018]. The time-bin deltas change sign, from +0.00022656 at rows99–499 to −0.00023541 at rows500–1,999, then small positive values later. See the detailed [results note](RESULTS_R1.md) and machine-readable [grade](grade_result_r1.json). This does not warrant a new B300 draft or a family-level reject. A fit-only scaling/bounding variant is the next useful local screen; existing causal-innovation and Kalman teacher/student handoffs remain in draft PRs [#14](https://github.com/VladKochetov007/WunderFunder/pull/14) and [#17](https://github.com/VladKochetov007/WunderFunder/pull/17).

### Seed-4 sensitivity follow-up

The frozen [seed-4 recipe](recipe_seed4_r1.json) repeats the same active-versus-zero comparison on the same fit components and F0 evaluation panel, changing only the stochastic seed. Active scored0.68037958 WP versus0.68039346 for zero control (Δ−0.00001387; conditional 141-component 95% interval [−0.00017548,+0.00012404]). Across seeds3 and4, the paired deltas are +0.00005163 and −0.00001387; their arithmetic mean +0.00001888 is descriptive, not a training-variability estimate. The feature effect is small and sign-unstable on this panel. The detailed [seed-4 result](RESULTS_SEED4_R1.md), [hash-bound prediction freeze](prediction_freeze_seed4_r1.json), [machine grade](grade_result_seed4_r1.json), and [execution/failure receipt](execution_receipt_seed4_r1.json) preserve the values and provenance. The first grade attempt failed after scoring because it expected absent `score.json` files; the same frozen pair then graded successfully with no score-dependent model change. This follow-up still does not reject the family or establish a robust benefit.

Fixed additive group states and fixed output smoothing are weak mechanisms on the measured screens. A separate literature-grounded direction is now being drafted: learn a causal, innovation-conditioned latent update/gain inside the recurrent teacher, keep raw inputs intact, and test teacher-to-student transfer. This changes the state-update mechanism and is not a new claim about the two local screens.

Predictions were frozen before grading with the official unchanged `global_wp_clip2_v1` metric and its F0 `need_prediction AND is_scored` mask. F0 has been inspected repeatedly; this is exposed development evidence, not clean validation or public-transfer proof. The one-seed pair gives a weak, mixed signal rather than a B300 allocation case. Keep the representation open and test a fit-only scaling or bounded activation locally before expanding the server plan; the existing retained-raw teacher/student direction is already in draft PR [#17](https://github.com/VladKochetov007/WunderFunder/pull/17).

## Resource and claim limits

Local work uses one admitted GPU worker, at most one CPU core for the fit, a4.5GB measured profile reservation within the current5GB per-task limit, no swap, and physical scratch at `/home/vlad/development/wunder/research_tmp`. Each phase is separately recorded. No submission ZIP, external upload, paid compute or B300 execution is part of this packet.

## Artifacts

- [`build_subset_manifest.py`](build_subset_manifest.py) selects complete non-evaluation components with a fixed seed and writes the trainer's validated source-qualified subset manifest.
- [`preflight_features.py`](preflight_features.py) and [`preflight_result_r1.json`](preflight_result_r1.json) bind the extractor implementation to its reference and serving-step behavior.
- [`profile_receipt_r1.json`](profile_receipt_r1.json) records actual GPU, RAM, finite-update, and OOM-counter evidence for the active feature shape.
- [`grade_pair.py`](grade_pair.py), [`grade_result_r1.json`](grade_result_r1.json), and [`RESULTS_R1.md`](RESULTS_R1.md) record the official paired grade, conditional component uncertainty, temporal breakdown and claim limits.
- [`recipe_seed4_r1.json`](recipe_seed4_r1.json), [`RESULTS_SEED4_R1.md`](RESULTS_SEED4_R1.md), the pre-grade [prediction freeze](prediction_freeze_seed4_r1.json), the [seed-4 grade](grade_result_seed4_r1.json), and the [execution receipt](execution_receipt_seed4_r1.json) record the second stochastic fit pair and its output-receipt repair.
- [`recipe_r1.json`](recipe_r1.json) freezes both arms, group membership rule, evaluator and compute bounds.
- Durable fit, prediction and scoring receipts are in the [campaign runtime](../runtime/README.md); weights remain under their run artifact directories and prediction arrays in the ignored cache.
