# E-516 result: target-wise simplex stacking of weak models

## Outcome

The fitted target-wise simplex assigned all weight to the C1 anchor in every target and outer fold. The affine intercepts made small prediction changes, but the pooled official endpoint declined. The registered recipe therefore fails its development gate and is closed for this fixed E-514 candidate pool and exposed chains.

This is not a test of model-weight merging, additional members, nonlinear residual correction, state gating, or newly trained candidates. It only tests this one prediction-level convex stack and intercept on cached outputs. E-514 had already exposed the same candidates and chains, and this follow-up was selected after observing E-514; all estimates are development evidence, not confirmation or a public forecast.

## Frozen comparison

- Anchor: `0.683226463` (`global_wp_clip2_v1`, 726,251 scored rows).
- Target-wise simplex: `0.682932549`; delta `-0.000293914`.
- Paired, complete-chain bootstrap 95% interval, conditional on the three fitted calibration solutions: `[-0.000824149, +0.000301622]`.
- All three folds selected anchor weights `[1, 0, 0, 0]` for both targets; positive weak-member mass appeared in `0` folds.
- Target 0: anchor `0.673356042`, stack `0.673196823`. Target 1: anchor `0.693096885`, stack `0.692668276`.
- Source-manifest SHA-256 `8de4dc6c1e2a9023510d3edbb3a46771579470bfb859ac50f451f7baa269d6fd`; chain-partition SHA-256 `e9e0cb9391789f54e34c1d02836d02c336c5d3a13bde6f9473196bbbc71da1c2`.
- Raw OOF prediction content SHA-256 `a9870ef179796f104ce20b14b01df5ed8d3c73416e52c9c308159819043e7785`; `.npy` artifact SHA-256 `ceca6104bda4082b397aa7b2363eef30792c083d6ea299cd167f98c1c2c95dc9`.

The paired interval crosses zero and the point estimate is negative. This calibration bootstrap conditions on the fitted fold weights and does not include training, candidate-selection, or public-transfer uncertainty. The source manifest, partition, and all fold-0 groups were previously inspected.

The post-score independent review verified the source and partition identities,
OOF byte digest, saved array, and official endpoint. Its recomputed anchor was
`0.6832264634983685`, stack `0.6829325494115299`, and delta
`-0.0002939140868385426`; these match the stored result within floating-point
rounding. The successful retry receipt binds the live cell code, source hashes,
row-group digest, outputs and exit status. Its precise cell runtime was not
instrumented during execution and is explicitly recorded as unavailable; the
Marimo output timestamp is a completion time, not a duration.

`calibration_weighted_mse` in the frozen JSON is an after-output-clipping
descriptive diagnostic. The registered coefficient fit itself minimizes the
pre-output-clipping weighted objective, and final held-out predictions are
clipped as specified. The diagnostic-label mismatch does not alter the fit or
the official endpoint; the result file is preserved unchanged.

The 123-row per-chain table and figure are descriptive checks only. Chain-level
WP is non-additive, and most deltas are effectively zero because fitted weak
weights were exactly zero; the pooled official endpoint remains the decision
statistic.

## Bandit decision

Stop this exact target-wise simplex/intercept recipe on this pool and exposed chains; do not tune the ridge, bounds, candidate scales, or member subset here. This result agrees with E-514 that these particular weak outputs are not recovered by convex blending. Keep the broader low-score/complementarity idea open only for a materially new, independently trained and ancestry-audited candidate cohort. A later test could ask whether weak predictions help as residual or state-conditioned features, but that needs new group-OOF inputs, input-only and shuffled-member controls, and genuinely unexposed confirmation. The E-515 target-0 error-location signal is a separate lead and cannot be used as proof that guided training will work.

## Human-readable artifacts

- [Target-wise fitted weights](plots/targetwise_simplex_weights.png)
- [Stack versus anchor and paired interval](plots/simplex_stack_vs_anchor.png)
- [Per-chain descriptive summary](plots/E516_CHAIN_SUMMARY.csv) and [per-chain delta figure](plots/simplex_per_chain_delta.png)
- [Machine-readable result](E516_RESULT.json)
- [Successful retry and provenance receipt](E516_SUCCESSFUL_RETRY_RECEIPT_20261007.json)
- [OOF prediction artifact](E516_OOF_PREDICTIONS.npy)
- [Preregistration and exposure limits](SIMPLEX_STACKING_PREREGISTRATION_E516.md)
- [Import failure and repair receipts](E516_IMPORT_PATH_FAILURE_20261007.json), [repair note](E516_IMPORT_PATH_REPAIR_20261007.md)
