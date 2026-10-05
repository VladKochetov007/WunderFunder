# Independent critic: target-mechanics EDA

Scope: review of `target_mechanics/eda_target.py`, `algebra_probe.py`, cohort and
saved raw/score evidence. No source files or fit artifacts were changed; no fit
was started.

## Admission checks

- The cohort independently rechecked against `cache/meta_valid.npz` and
  `cache/chains.npz`: all 40 row-group/sequence/class/component identities
  match; no selected group is in the protected holdout; all 40 components are
  distinct; each diagnostic class has 6 fit and 4 test sequences. The checked
  groups are complete 20,000-row sequences.
- Current/past features are causal within each row group. The lag indexing
  clamps at row zero, so it cannot cross sequence boundaries. `events[t]`
  compares current row `t` with `t-1`; the target-change array is shifted so its
  element at `t` also denotes `y[t] != y[t-1]`. Offset `+1` is correctly marked
  future-only diagnostic. `event_hold` uses each sequence's own last event and
  the predictor at that event row; it never imports labels from the previous
  row.
- Fit rows are `128, 148, ...`; fit uses no validation scoring mask. Test
  scoring uses same-row `need_prediction & is_scored`, and the official helper
  clips both target and prediction and applies `abs(clipped target)` weights.
  The recorded 26,990 test rows equal the scored-row totals in raw evidence.
  The official metric implementation and data overview confirm this mask and
  pooled-row definition.
- Target-derived class is only used to stratify and report this diagnostic
  cohort. Models fit across all classes and never receive that class. Selection
  does nevertheless condition test composition on this label-derived metadata,
  so the pooled score is a diagnostic cohort score, not a naturally sampled
  overall validation estimate.

## What is and is not established

The EDA credibly describes strong target persistence, anticorrelation, small
per-sequence supports, and that changes often coincide with changes in the two
selected observed price-like columns. It does not establish a sufficient event
cause, physical meaning for anonymized columns, a universal finite alphabet,
or that a deployed model should hold its prediction between events. The hard
causal event-hold score (.5056 vs direct HGB .5347 on the same 26,990 scored
rows) only falsifies the exact HGB-output-freezing rule in this sample. It does
not test a complete recurrent predictor or a learned event-conditioned state
update. In particular, weak HGB performance is not evidence that a GRU is
weak, nor that an alternate head beats the campaign incumbent.

There is one important evidence-coverage gap: `structure(y, raw)` calculates
target-change/event alignment over all 20,000 rows, including warm-up and rows
that are not scored. This is valid descriptive sequence evidence, but the near
necessity figures should not be read as the event law on the competition's
scored population. The bounded falsifier is to recompute the change/event
contingency on the same 40 sequences, indexing transitions by their destination
row and retaining only `need_prediction[t] & is_scored[t]`; report counts by
class and a sequence/component bootstrap interval. If alignment materially
weakens or varies across components, do not use it as a gating prior. The
existing event-hold predictor is already scored on the correct mask, so this
gap does not invalidate its .5056 comparison.

The four test sequences per class also make sequence-level uncertainty large;
class 2 has only 1,365 selected scored rows (including a 160-row sequence).
Rows within a sequence are highly dependent, so row-level standard errors are
misleading. Keep all reported predictor scores explicitly exploratory and
require additional components plus paired component-level comparisons before
making architecture claims.

## Objective equivalence and useful inductive bias

For a selected-row target, let `Y=clip(T,-2,2)` and `w=abs(Y)`. Ignoring the
small training floor, minimizing conditional weighted squared error
`E[w (f(X)-Y)^2 | X]` gives

`f*(X) = E[wY | X] / E[w | X]`.

This is also the scalar predictor that optimizes weighted Pearson correlation
up to an affine transformation (then clipped), because the metric is ordinary
Pearson under the target-tilted measure proportional to `w`. Therefore a joint
or per-target conditional distribution head decoded to that ratio does not
create a new pointwise scoring objective. Its gain must come from an explicit
inductive bias or better optimization, not the word “distribution.” If it is
trained with `w * -log q(Y|X)`, it models the tilted distribution and its
ordinary expectation is the same ratio; that still does not alone demonstrate
an advantage over direct weighted-MSE regression.

Specific implementation cautions:

- The current direct and factor HGB controls use `abs(Y)+1e-3`, not the exact
  metric weight. This is a harmless stated stabilizer for a cheap baseline, but
  not an exact empirical WMSE minimizer; zero-target rows get nonzero training
  influence even though they have zero score weight.
- Rotating to sum/difference, or multiplying separately estimated sign and
  magnitude, is not a distinct causal model. The latter loses pooled score by
  about .030 here. A joint distribution over two targets only adds capacity
  sharing; it needs a restriction such as persistent latent-state dynamics,
  train-defined quantization with an escape residual, or an explicitly coupled
  transition prior to make a falsifiable architecture claim.
- The finite supports are per-sequence observations, not proof of a fixed
  global alphabet. Quantization must have a residual/unknown-value path and be
  tested on components with unseen amplitudes.

## Decision and bounded next comparison

Treat this EDA as a useful mechanism-generating diagnostic, not evidence that
it closes the gap to the 0.7000 goal. Do not infer a win against the incumbent
or GRU from these HGB results. Once the strong complete predictor diagnostics
are available, use fixed component-disjoint sequences and paired seeds to
compare its direct weighted-mean heads against exactly one structured arm:
either a causal latent run-state/update head (soft, never hard output holding),
or a train-defined joint support distribution plus residual. Match trunk,
training rows, optimizer budget, and readout; report per-target global WP,
paired component deltas, calibration/support escape rate, and score by observed
event/no-event rows. Reject the structured arm if it does not beat the matched
direct head across components, if a shuffled-event sham matches it, or if its
gain is only label-class composition. Before using event gating as a prior,
run the scored-destination transition check above.
