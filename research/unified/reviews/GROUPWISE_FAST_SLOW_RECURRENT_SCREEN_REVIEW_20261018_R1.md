# Groupwise fast/slow recurrent screen review — 18 October 2026

## Verdict

**The matched recurrent screen is internally consistent, but its point increase is too small and uncertain to justify a separate B300 draft. Keep the feature family open.** The result is relevant context for the retained-raw Kalman teacher/student path in [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17). It does not test the projected-innovation implementation in PR #14.

## Comparison and execution audit

Both arms used the same 32-component fit manifest, initializer checkpoint, initial parameter hash, 168 selected training sequences, seed/order, normalization, and 210 completed FP32 updates. The only treatment difference was whether the 39 appended groupwise states contained their computed values or zeros. Both checkpoint and prediction sidecars bind the fixed evaluation groups; the feature-only exports cover the same 468 F0 groups.

The official grade uses `global_wp_clip2_v1`, the fixed F0 evaluation membership, the cached `is_scored` mask, and the post-warmup prediction rows. The feature preflight reports 39 finite channels, NumPy/Torch agreement within `9.54e-7`, exact zero prefix change after future-row perturbation, exact reset repeat, and zero warmup outputs. The independent audit found no result-changing mismatch in the recipe or grade.

The actual-shape profile peaked at 4.052 GB under a 4.5 GB cap and recorded 6,060 `memory.high` events; worker max/OOM/kill deltas were zero. Both fits and prediction attempts completed without worker OOM/max events. Keep further work serialized and within the existing caps. The profile's high-water pressure is a reason to retain admission checks, not evidence of an OOM or a scientific failure.

## Result and limits

| Endpoint | WP |
|---|---:|
| Active 39 states | 0.67986413 |
| Equal-width zero control | 0.67981250 |
| Paired difference | +0.00005163 |

The paired bootstrap over 141 connected evaluation components gives a conditional 95% interval of `[-0.00013143, +0.00027018]` and `p(delta <= 0)=0.3125`. The score files and grade receipt agree with the reported values. The interval conditions on two fixed checkpoints and repeatedly inspected F0; it omits fit-seed, fold, selection, and public-transfer uncertainty. Position-bin effects are descriptive and change sign.

This is one exposed-fold, one-seed, 32-component incremental fine-tune on a common parent, not a scratch-trained teacher or an independent replication. The result neither demonstrates a useful gain nor rejects recurrent Kalman features. The one-seed screen also did not test whether fit-only scaling or bounding the appended features changes their use. A scaled/bounded follow-up remains a possible local experiment if its expected information value justifies another fit pair.

## Sources

- [Experiment packet](../../campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/README.md) and [detailed result](../../campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_R1.md).
- [Grade result](../../campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/grade_result_r1.json), [profile receipt](../../campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/profile_receipt_r1.json), and durable controller attempts under `research/campaign_2026-10-18/online_features_r1/runtime/attempts/`.
- Parent hypothesis: `H-GROUP-FAST-SLOW-INNOVATIONS-20261018-R1`. Recurrent interaction hypothesis and experiment: `H-KALMAN-GROUP-RECURRENT-INTERACTION-20261018-R1` and `E-KALMAN-GROUP-RECURRENT-INTERACTION-20261018-R1`.

