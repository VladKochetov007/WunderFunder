# Lagged-label proxy screen R1 results

## Observed result

The single weighted-ridge fit used 32 source-qualified train components and
63,680 sampled rows per target. At each current step it predicted the clipped
previous-row label from the 112 current features and their one-step change.
The independent offline grade used 32 distinct, component-disjoint F0
evaluation sequences, predicting target rows 99–19,998 from the following
feature rows. It produced 636,800 two-target predictions and 62,016 masked
timestep positions.

| Quantity | Result |
|---|---:|
| Previous-row-label proxy WP | 0.5216794 |
| t0 WP | 0.5335995 |
| t1 WP | 0.5097594 |
| Conditional 32-component bootstrap 95% interval | [0.4488578, 0.5768614] |
| Independent regrade | Exact match; absolute score difference 0 |

The bootstrap resamples the fixed 32 evaluation components and conditions on
this one fitted proxy. It does not cover fit-seed variation, data-selection
uncertainty, target maturity, or transfer. F0 is exposed development data.
The zero-output score is a metric sanity control; it is not a competition
baseline. This score also must not be compared numerically with model endpoints:
it is a different one-row-shifted prediction task.

## Interpretation

Current state and its one-step change contain a strong signal for predicting a
previous-row target label on this exposed panel. This supports the feasibility
of a *feature-derived pseudo-label*. The docs do not reveal the target horizon
or when a previous-row label matures, so this is not evidence that a realized
label is available online. No proxy-driven update was applied, and no served
endpoint gain, test-time learning effect, competition improvement, or public
transfer has been measured.

The next bounded screen is a causal, proxy-driven expert gate using existing
frozen TL3 and R3 prediction caches on the same 32 components. At step `t`, the
gate will compare the previous-step expert outputs with the proxy generated
from `x[t]`, update state reset per sequence, and weight the current expert
outputs. Evaluation labels/masks remain grader-only. This tests whether the
proxy is useful for an actual endpoint; it does not resolve target maturity.
The prior fixed TL3/R3 blends underperformed TL3, so the dynamic gate must be
graded against TL3 and the fixed 75/25 control.

## Execution and evidence

The full fit completed in 5.03 seconds with a 1,344,704,512-byte worker-cgroup
peak under a 2 GB reservation. The full grade completed in 5.04 seconds at
663,388,160 bytes under the same reservation. The independent regrade used
336,158,720 bytes under 1.5 GB. All three attempts completed without worker or
parent OOM-event increments; no GPU was used. The earlier under-sized metadata
preflight OOM is preserved separately and is not erased by these clean runs.

See the [frozen recipe](recipe.json), [fit receipt](fit_receipt_r1.json),
[grade receipt](grade_receipt_r1.json), [independent regrade](independent_grade_r1.json),
[execution receipt](execution_receipt_r1.json), and [one-group input profile](profile_receipt_r1.json).
Model weights and prediction arrays remain in physical scratch, with their
hashes and paths recorded in the grade/execution receipts.
