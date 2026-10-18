---
kind: result
status: complete
created_utc: 2026-10-18T00:02:06Z
updated_utc: 2026-10-18T00:02:06Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-18/online_features_r1/CONTRACT.json
  - research/campaign_2026-10-18/online_features_r1/DATA_AUDIT.md
  - research/campaign_2026-10-18/geometry_variations_r3/geometry_r3_grade.json
  - research/campaign_2026-10-18/online_features_r1/prediction_geometry/receipt.json
---

# Exposed F0 prediction geometry diagnostic

## Question or claim

How do five already-frozen predictors and their corrections relate to the
parent on source-qualified fold0, including the four position ranges declared
before looking at the labels? This is a diagnostic geometry report for later
independent review. It is not a clean evaluation or a deployment decision.

## Mechanism and method

The panel has 468 evaluation sequences, 141 source-qualified components, and
808,863 rows selected by the official target-cache mask after warmup position
99. Every prediction was aligned to `evaluation.valid` through sidecar
`row_group` IDs. The evaluator used `global_wp_clip2_v1`, clipped labels and
predictions at ±2, and the original weight `abs(clipped label)`. It did not
send labels or the mask to a predictor.

The frozen set was the F0 parent, R2 full geometry, R2 `m0p50` fixed inference
alias, R3 moments-only, and R3 dictionary-only. There are 37,742 selected rows
in `[99,1000)`, 159,998 in `[1000,5000)`, 198,988 in `[5000,10000)`, and 412,135
in `[10000,20000)`. Bin panel coverage and per-target weighted mass are:

| Position range | Selected rows | Sequences | Components | Weight mass t0 | Weight mass t1 |
|---|---:|---:|---:|---:|---:|
| `[99,1000)` | 37,742 | 435 | 136 | 25,922.2894 | 28,451.1414 |
| `[1000,5000)` | 159,998 | 466 | 140 | 110,252.1998 | 114,278.9341 |
| `[5000,10000)` | 198,988 | 468 | 141 | 142,803.5798 | 143,659.0496 |
| `[10000,20000)` | 412,135 | 468 | 141 | 282,265.5731 | 288,925.6860 |

For each target, the receipt contains weighted covariance and correlation for
clipped candidate prediction vs parent, parent-relative correction vs parent,
and correction vs label. It also reports the correlation of correction and
label residuals after each was projected on the fixed parent using a pooled
F0 weighted intercept and slope. The pooled coefficients are reused in every
bin; there are no bin-specific refits.

## Evidence and interpretation

All five recomputed pooled scores match the frozen independent grade within
`2e-6` (the observed differences are at floating-point roundoff). Pooled
official scores are primary; the position bins are diagnostics only.

| Predictor | Pooled WP | t0 | t1 | Δ WP vs parent | `[99,1000)` | `[1000,5000)` | `[5000,10000)` | `[10000,20000)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Parent | 0.681226513 | 0.674220742 | 0.688232284 | — | 0.677936 | 0.688169 | 0.650104 | 0.694939 |
| R2 full geometry | 0.680449904 | 0.673181545 | 0.687718263 | -0.000776609 | 0.677841 | 0.687543 | 0.649232 | 0.694095 |
| R2 `m0p50` alias | 0.681422110 | 0.674531852 | 0.688312368 | +0.000195597 | 0.677910 | 0.688286 | 0.650306 | 0.695175 |
| R3 moments-only | 0.680868024 | 0.673798572 | 0.687937477 | -0.000358489 | 0.678242 | 0.687753 | 0.649690 | 0.694574 |
| R3 dictionary-only | 0.680616866 | 0.673417560 | 0.687816172 | -0.000609647 | 0.677962 | 0.687630 | 0.649350 | 0.694315 |

The per-bin `t0 / t1` scores, in the same bin order, are:

| Predictor | `[99,1000)` | `[1000,5000)` | `[5000,10000)` | `[10000,20000)` |
|---|---:|---:|---:|---:|
| Parent | 0.642964 / 0.712908 | 0.662912 / 0.713426 | 0.646669 / 0.653539 | 0.696885 / 0.692992 |
| R2 full geometry | 0.643476 / 0.712207 | 0.662459 / 0.712626 | 0.645254 / 0.653211 | 0.695685 / 0.692504 |
| R2 `m0p50` alias | 0.642581 / 0.713240 | 0.663012 / 0.713559 | 0.647122 / 0.653489 | 0.697256 / 0.693094 |
| R3 moments-only | 0.643690 / 0.712794 | 0.662774 / 0.712732 | 0.645974 / 0.653407 | 0.696397 / 0.692750 |
| R3 dictionary-only | 0.643363 / 0.712561 | 0.662466 / 0.712794 | 0.645558 / 0.653141 | 0.696005 / 0.692626 |

The scale alias's within-panel WP difference ranges from `-0.000026` in the
earliest bin to `+0.000236` in the latest. This describes the fixed predictions
on this exposed panel only; it does not establish a gradient effect, explain a
cause, or support generalization. No inference is drawn from the sign changes
for the fitted R2/R3 variants either.

The candidates remain very close to the parent in weighted prediction geometry:
pooled prediction-parent correlations are about `0.9996`–`0.9999`. Their
corrections have weighted parent-projection R² from `0.573` to `0.748` across
the two targets and four non-parent candidates. After the fixed pooled-F0
parent projections, pooled correction-residual / target-residual correlations
are small, from approximately `-0.039` to `+0.037`. These are descriptive
same-panel relationships and use exposed labels; they are not evidence that a
correction is redundant or that residual signal transfers. Full pooled and
per-bin covariances, correlations, and projection coefficients are in
[receipt.json](receipt.json).

## Limits and validity

Fold0 has been inspected repeatedly and is not a fresh holdout. This report
adds no bootstrap, model fit, prediction replay, alpha search, or coefficient
selection. The R2 scale alias is an inference-only derivative of the R2
checkpoint, not an independent fit. Prediction hashes and sidecar hashes are
reused from the frozen grade rather than recomputed. Unknown TL3 training,
teacher, initializer, and sequence ancestry remains unknown; TL3 was not added
to this grid. No venue, clock, event, market cause, or online-learning rule is
inferred from these anonymous sequence positions or labels.

## Decision and next step

No SHIP/KILL judgment is made. Pass the receipt to the independent MFT skeptic
for interpretation alongside the experiment's causal contract and ancestry.
Treat any position pattern as an exposed-F0 diagnostic, and retain the
component-clean F0 parent as the comparison reference.

## Source records

- Campaign contract and online data limits:
  [CONTRACT.json](../CONTRACT.json) and [DATA_AUDIT.md](../DATA_AUDIT.md).
- Frozen geometry grade:
  [geometry_r3_grade.json](../../geometry_variations_r3/geometry_r3_grade.json).
- Official metric implementation: [gmetric.py](../../../../src/wnn/gmetric.py).
- Reproducible calculation and machine receipt:
  [geometry_diagnostic.py](geometry_diagnostic.py) and [receipt.json](receipt.json).
