# All-channel causal innovation clock

## Question

Can a small diagonal Kalman filter over the 112 raw inputs produce a useful
sequence-local change clock for a recurrent model? This first screen only
checks that the feature is causal, stable and nondegenerate in fit-side data.
It does not fit an endpoint or score a target.

## Frozen recipe

[recipe.json](recipe.json) fixes a 32-component sample drawn only from
`fold0.fit.train`, one complete sequence per component. It centers and scales
each channel from the first 99 observed rows, computes pre-update standardized
innovations with fixed diagonal `Q=0.001`, `R=1.0`, averages clipped squared
innovation across channels, and updates a bounded CUSUM. The candidate scalar
is `log1p(CUSUM)`; original input channels remain intact. All estimator state
resets at sequence boundaries. The feature reader requests only `seq_ix` and
the 112 raw input columns.

The source footer and split hashes are checked before opening Parquet metadata.
The measured 1.5 GB reservation comes from the successful metadata-only and
one-row-group source-reader profiles in the [source audit](../../../unified/operations/P2_INNOVATION_CLOCK_SOURCE_AUDIT_20261018_R1.md)
and [runtime profile records](../causal_label_proxy_r1/profile_receipt_r1.json).
The old 1.0 GB unbounded metadata attempt remains a separate incident and is
not repeated.

## Results

The bounded pass completed on 32 fit-side components / 640,000 input rows.
Average component-level innovation RMS rose from 1.792 at rows 99-499 to
2.635 at rows 10,000-19,999. Innovation distributions differed consistently
by semantic group: `i1_ask_price_like` was highest through later bins and
`i0_trade_price` remained among the lowest. The scalar CUSUM's median
component 95th percentile reached its `log(21)` cap by rows 500-2,000; its
upper tail loses resolution, while component and time variation remain.

This is a feature-distribution result with no target association or endpoint
score. It may reflect warmup-scale mismatch, later volatility changes, or
sequence-specific regimes. The next representation to screen is a group-aware
fast/slow innovation vector that preserves differences between the 13 semantic
channel groups. An independent [Data Sentinel review](../../../../unified/reviews/INNOVATION_CLOCK_FEATURE_SCREEN_REVIEW_20261018_R1.md)
confirmed source membership, input-only projection, causality/reset checks,
resource receipt and summary hashes. It found a wording issue in the original
machine result's top/bottom group labels; the
[correction addendum](review_correction_r1.json) preserves the per-bin result
without rewriting the original artifact. The [result](result_r1.json) and
copied [feature-only receipt](feature_only_receipt_r1.json) preserve measured
values and provenance; detailed tables stay on physical scratch. No submission
or new B300 draft follows from this diagnostic. Broader causal-innovation and
Kalman teacher/student directions already exist in draft PRs #14 and #17.

## Source records

The screen implementation is split into [innovation_features.py](innovation_features.py)
and [screen.py](screen.py). It is registered as
`H-ONLINE-INNOVATION-CLOCK-20261018-R1` /
`E-ONLINE-INNOVATION-CLOCK-FEATURE-20261018-R1`, descending from H-240.
