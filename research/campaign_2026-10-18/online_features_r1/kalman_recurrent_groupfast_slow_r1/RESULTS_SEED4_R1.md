# Recurrent groupwise Kalman-state seed-4 sensitivity screen

**Status: completed paired seed-4 screen. The active feature effect changes sign relative to seed 3 and remains much smaller than evaluation and fitting uncertainty. Keep the feature family open; this result does not support a stable gain.**

## Frozen question and comparison

Does the active-minus-zero effect for the same 39 causal groupwise fast/slow Kalman innovation-energy states persist under a second optimizer seed on the same exposed F0 panel? The seed-4 recipe changes only stochastic training randomness from the seed-3 pair. It retains the same `warm_mechanism_f0_s3_c_r3@ep1` parent, 32 complete source-qualified fit components, optimizer, update count, checkpoint policy, feature code, precision, and all-zero equal-width control.

The fit uses 102 training row groups and 66 development groups from the same 32 components. Both models are incremental fine-tunes of a shared parent; this is not a scratch-trained teacher, independent data replication, fold replication, or public-transfer test. The original [seed-3 result](RESULTS_R1.md) and this repeat use the same repeatedly inspected F0 evaluation groups, so seed repetition does not make F0 fresh.

## Prediction freeze and grade

The active and zero-control prediction exports were checked as finite FP32 arrays of shape `[468, 20000, 2]`, with matching evaluation membership and pinned checkpoint, prediction, and sidecar hashes. The [prediction freeze](prediction_freeze_seed4_r1.json) was written before target or mask arrays were opened (`targets_read=false`, `masks_read=false`). The grader verifies that freeze and those hashes before loading the official cached targets/mask.

The first grading worker completed the comparison calculation but failed while assembling the output receipt: this `--no-eval-during-fit` recipe has no per-run `score.json`, while the original grader tried to hash one. It wrote no result and emitted no score. The grader was repaired to identify each endpoint by its frozen checkpoint and prediction hashes, then the exact same pair was graded successfully. Both attempts were after the prediction freeze; no model or prediction was changed in response to the failed attempt. The [execution receipt](execution_receipt_seed4_r1.json) preserves both attempts and the failure diagnosis.

## Official metric results

Scores use unchanged `global_wp_clip2_v1`, the fixed F0 evaluation groups, cached `need_prediction AND is_scored`, and the row-99 scoring boundary.

| Endpoint | WP | Target 0 | Target 1 |
|---|---:|---:|---:|
| Active 39 states, seed 4 | 0.68037958 | 0.67320791 | 0.68755126 |
| Equal-width zero-state control, seed 4 | 0.68039346 | 0.67317990 | 0.68760701 |
| Active minus control | **−0.00001387** | +0.00002801 | −0.00005575 |

The paired bootstrap resampled 141 connected evaluation components over 2,000 draws. Its conditional 95% interval for the WP difference is `[−0.00017548, +0.00012404]`; `p_le_0=0.5805` is the fraction of bootstrap draws at or below zero, not a formal hypothesis-test p-value. This interval conditions on the two fixed seed-4 checkpoints and exposed F0 components; it excludes fit-seed, fold, feature-selection, and public-transfer uncertainty.

Post-hoc position-bin deltas are mixed: `−0.00045064` at rows 99–499, `−0.00007294` at 500–1,999, `+0.00006737` at 2,000–9,999, and `−0.00004715` at 10,000–19,999. These bins are descriptive and not separate confirmatory tests.

## Across-seed reading

| Fit seed | Active WP | Zero-control WP | Paired delta | Conditional component 95% interval |
|---:|---:|---:|---:|---:|
| 3 | 0.67986413 | 0.67981250 | +0.00005163 | [−0.00013143, +0.00027018] |
| 4 | 0.68037958 | 0.68039346 | −0.00001387 | [−0.00017548, +0.00012404] |

The arithmetic mean of the two paired deltas is `+0.00001888 WP`; this is descriptive only. The sign flips, both intervals span zero, and absolute endpoint scores move about `0.00058 WP` between optimizer seeds while the treatment effect is at most `0.000052`. Two seeds on one exposed fold are not enough to estimate fit-seed variability reliably. This weakens the claim of a stable increment but does not reject the broader conditional/recurrent Kalman family.

## Resources and provenance

Both fits, both feature-only prediction exports, and the successful grade ran sequentially under the current controller, one CPU core per worker, no swap, and bounded reservations. Fit peaks were 2.62 GB and 2.58 GB RAM; prediction peaks were 1.58 GB and 1.46 GB RAM; grade peak was 0.58 GB. Every successful attempt had zero worker and parent memory-high, max, OOM, and kill deltas. The detailed [execution receipt](execution_receipt_seed4_r1.json) binds attempt IDs, commands, source hashes, limits, and terminal status.

- Frozen recipe: [recipe_seed4_r1.json](recipe_seed4_r1.json), SHA-256 `a6adde488489f50862406f77f1224eb94f745540ee68b9f986e77bc7c579fdda`.
- Fit membership: [fit_subset_manifest_seed4_r1.json](fit_subset_manifest_seed4_r1.json), SHA-256 `658aac191be4973c36575b638dc32bb1def7d5543b6bdd06edd6b3931e280f78`.
- Prediction freeze: [prediction_freeze_seed4_r1.json](prediction_freeze_seed4_r1.json), SHA-256 `5a2d62de6974e2b207265e492dd6aca3117210f203b0abebc1f458d75dc92451`.
- Machine grade: [grade_result_seed4_r1.json](grade_result_seed4_r1.json), SHA-256 `e1751b2f83e87d543dc441a483d5714f9cde06bd66c6951839ecf111d1a6717b`.
- Successful grade contract: [grade_contract_seed4_retry_r1.json](grade_contract_seed4_retry_r1.json), SHA-256 `e0e982e4c5a21fd36dc28306c6f710d48f53a524efc2476abadc98440d2bef6f`.
- Shared initializer SHA-256: `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`.
- Active/control checkpoint SHA-256: `158893b501ad67642f9deed3dbf87639374a11b3684aed14fc0d69566e224e93` / `981ddddfbd7552622dc5144ba2e860f5c7e859f2c4a2f1b255b26542eade08a7`.
- Active/control prediction SHA-256: `132df658a7767a48df4922d35230ebce07bd3b221da9a80e0d8898a72c337139` / `7f41ad6f33bf6c424f679031895869b4420c2fc4f99706be631b6f77ff65511f`.

## Decision

Do not promote the 39 appended states as a proven quality improvement, and do not close the feature family from this two-seed result. The next *large* Kalman experiment should change the mechanism: learn a causal, innovation-conditioned latent update/gain inside a recurrent teacher while preserving raw inputs, then test whether its supervision transfers to matched GRU students. Direct output smoothing is a different treatment and already lost materially; fixed additive summaries have not shown stable value. The larger existing [PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17) remains the local-trend teacher/student handoff. A separate learned-gain draft is being prepared because it changes the recurrent state-update mechanism rather than adding another fixed feature block. No B300 fit or competition score is claimed here.
