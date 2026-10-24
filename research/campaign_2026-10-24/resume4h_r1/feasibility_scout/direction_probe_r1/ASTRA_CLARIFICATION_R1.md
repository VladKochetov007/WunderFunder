# Direction probe clarification R1

This note records the 2026-10-24 clarification to Astra's [direction-probe proposal](../../astra_fundamentals_r1/DIRECTION_PROBE_RECIPE_R1.md). It narrows the executable estimand without changing the original proposal file.

## Frozen projection point

All nuisance projections are evaluated at alpha 0. Every candidate shares the same raw parent base `p`. The four calibration tangent columns are the right derivatives of clipping at `p` in directions `+1`, `-1`, `+p`, and `-p`. The A tangent is also differentiated through that same `p`. Calibration and calibration-plus-A spans are standardized in weighted variance units, then ranked from the eigenvalues of their weighted correlation Gram matrix. Retain only `lambda > lambda_max * 1e-10`; report the full spectrum, dropped numerical-zero columns, and materially negative eigenvalues.

At alpha 0.1, report the candidate's right derivative and clip-state diagnostics only. Do not project at that endpoint. This avoids changing the estimand or introducing a second projection basis.

## Arithmetic and official scorer branches

Analytic derivatives use real arithmetic over the saved float32 values. The already-graded alpha-0.1 endpoint is reproduced separately with the historical float32 operation `parent_f32 + np.float32(0.1) * (candidate_f32 - parent_f32)`. The two target derivatives use their own clipped-label weights and are averaged only after each target's score gradient is calculated.

Match `global_wp_clip2_v1` gates with `_EPS = 1e-8`: total weight is inactive only below epsilon; target and prediction standard deviations are inactive at or below epsilon. Report distances to all gates and classify exact-boundary behavior; do not use a quotient derivative at a prediction-SD gate crossing. Apply the right derivative of the final correlation clip at exact `rho = +/-1`. Projection decomposition is reported only on the regular linear score branch.

## Component evidence and limits

For every endpoint and target, save additive weighted totals, all first moments, and the full cross-product matrix. Also save stable centered component moments for scoring and bootstrap recomputation. A fixed 2,000-draw connected-component bootstrap resamples these component tables and reports rank changes. The synthetic check compares resampled aggregation with direct row-level recomputation.

F0 A lag-1 and lag-16 are one exact prediction endpoint and appear once, with the lag-16 identity recorded as an alias. F1 A is a separate source-qualified fold and parent. Astra's proposal made F1 conditional on its separate preflight; this packet amends that scope because the F1 R2 preflight and fixed-endpoint grade already completed and passed before this probe. Both verified cells are mandatory in this machine recipe, which fails closed on any input or ancestry mismatch. Both evaluations are already exposed. Any intervals are conditional diagnostics, not new confirmation. Evaluation-label-weighted coefficients are diagnostic only; no alpha search, coefficient export, endpoint selection, new fit, or public-score claim is authorized.
