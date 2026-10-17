# Why near-constant labels do not imply constant predictions

This note is a mathematical interpretation of the supplied metric and the
predeclared R2 diagnostics. No model, coefficient, calibrator or density estimate
is fitted here. Predictions are fixed historical artifacts; all local panels are
exposed. The decomposition is ours, not a benchmark claim borrowed from a paper.

## Two different questions about the old clock

The R1 census found 1,253 need-eligible target-change rows outside the conjunction
of changes in `i0_p0` and `i0_p11`, among 249,600,000 total rows and 33,278,744
need-eligible target-change rows. These are about 5.02 parts per million of all
rows and 37.65 parts per million of target-change rows. Dividing a joint two-target
change count by the number of individual scalar labels would answer a different
question. The user is right that these counts alone do not show practical failure.

Among valid's 3,528,273 selected rows there are only92such change rows. Because
each of two clipped-label magnitude weights is at most2, their combined weight
share is at most `92*4/4,895,308.008`, approximately0.00752%. This is a bound on
the weight of those transition rows, not a bound on score change: correlations
depend on centered moments and prediction variance. Nor does it cover persistent
damage after a missed transition. The R2 census measures that separately by
retrospective reconstruction of labels from input-clock boundaries.

The corrected claim is therefore: the clock has near-perfect recall for observed
target changes, but is not exact. Its precision is only about62–63% on need-eligible
train/valid rows: many clock events do not change either target. It is not a
one-to-one label generator. Neither the old census nor the
full-vector sample establishes that a hard or soft event-driven architecture
would improve or hurt a model. A changing input is not necessarily useful input.

## Information can arrive while the realized label stays fixed

Suppose a fixed latent quantity Z produces a constant target over an interval,
while successive input observations reveal noisy evidence about Z. The realized
label does not change, but a causal estimate E[Z | observations through t] should
change as information arrives. Holding its first estimate discards that evidence.
Conversely, if later observations contain mostly irrelevant variation, a model
may fluctuate unnecessarily and a selective emission rule could help.

These alternatives make opposite predictions for existing-model output holds.
They cannot be distinguished by label persistence alone. They also distinguish:

- reading every input row;
- updating all, some or none of the hidden state;
- recomputing a head;
- emitting a new value versus repeating an earlier prediction.

The R2 interventions change the last item using saved predictions. They cannot
measure runtime savings or the quality of a model whose hidden updates are skipped.

## Exact weighted variance decomposition

For one target, let y and p already be clipped to [-2,2]. Let w=M|y|, with M the
official selected-row mask, and use normalized weighted expectations E_w. Split
each sequence into maximal runs with exactly constant clipped y; boundaries are
defined separately for each target. Within a run r, weights of selected rows are
constant, so its selected-row prediction mean pbar_r is also its weighted mean.

Set e_t=p_t-pbar_r. Within every run, sum(w e)=0 and y is constant. Consequently:

```
E_w[e] = 0
Cov_w(y, p) = Cov_w(y, pbar)
Var_w(p) = Var_w(pbar) + E_w[(p-pbar)^2]
WP(pbar) = WP(p) / sqrt(1 - within_run_variance / Var_w(p))
```

The last equation requires nonzero variances; it gives an increase for positive
baseline correlation, not for a negative correlation. It is exact before the
implementation's float32 rounding and floating-point accumulation. It can be
checked numerically against the canonical scorer. Averaging raw predictions and
then clipping is different and does not preserve this identity.

**pbar is an oracle diagnostic.** It requires label-defined boundaries and later
predictions within the same run. It is not causal, not a deployable candidate,
not a Bayes ceiling, and not evidence that a causal filter can realize the gap.
The gap measures the penalty associated with temporal fluctuations of this
specific prediction array relative to this retrospective partition.

The distinction matters even if oracle averaging helps and every causal filter
hurts: a future-informed average can denoise useful progressive estimation,
while any causal average lags that estimation. Such a result would favor a
better learned state/readout or teacher target, not a frozen-output wrapper.

## The existing loss already contains a persistence penalty

Within a constant-label run, for the same evaluated prediction variable:

```
sum_t w (p_t-y)^2 = W_run (pbar_run-y)^2 + sum_t w (p_t-pbar_run)^2.
```

Thus magnitude-weighted squared error already penalizes both an incorrect run
mean and within-run prediction fluctuation. Adding a generic consistency loss is
not discovering a missing term; it changes its relative weight, which may suppress
useful updates or worsen optimization. A genuinely distinct proposal must state
what changes: causal state factorization, training-only teacher smoothing,
nonuniform uncertainty weighting, chunk normalization, or optimization horizon.
The repository already has temporal bootstrapping/reanchoring in
`src/wnn/training/losses.py`; a proposal must compare with those exact predecessors.

This identity is about a fixed run's additive squared error. The code's raw-output
training loss need not clip predictions like the official scorer; variable
batch/chunk denominators also alter relative update weights. Neither issue is a
new proven bug or automatic improvement: old objective comparisons must be kept.

## Causal model and evaluation diagram

```text
permitted input history ──> recurrent state ──> emitted prediction ──> scorer
                                                        labels ──> scorer
permitted training labels ──> fitted parameters ──> state/readout
```

Predictions do not cause the fixed competition labels. The first line is a
computation graph; it makes no assertion about physical market causality.

## Decision table fixed before reading R2 scores

| Observation | Interpretation that becomes more plausible | What still does not follow |
|---|---|---|
| Exception weight and persistent reconstruction distortion tiny | Keep the approximate clock; rare violations are a low-priority explanation for the performance gap | Exact equality or causal sufficiency of the selected prices |
| Holding outputs improves several frozen models on matched coverage | Existing emissions may be too reactive relative to this clock | Hidden-state skipping helps, public score improves, or an optimal threshold was found |
| Holding hurts but causal smoothing helps | Graded updates may be preferable to hard holding | One selected smoothing constant is a validated candidate |
| Oracle averaging helps, causal holding and smoothing hurt | Later inputs/estimates may add useful information; future averaging removes noise with unavailable information | The oracle gap is a realizable causal gain or all consistency mechanisms fail |
| Oracle gap already tiny | Within-run output fluctuations are a weak explanation for these models | The state representation or training has reached a Bayes limit |
| Effects disagree across models/folds | Recipe/exposure/calibration interaction matters | A universal architecture conclusion |

Report all prespecified transforms and their exact masks. Do not fit a gate,
choose a serving filter or describe the best of these diagnostics as confirmed
quality. Source-qualified component uncertainty, when available, remains
conditional on the existing predictions and their fitting/selection history.
