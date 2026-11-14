# Eight training ideas for the causal event-clock regressor

Everything below assumes one change per run, evaluated on the same regime-shifted block, 3 seeds, and a
**noise floor of 0.002**. That floor is the important number: seed ensembles correlate 0.99, so the blend
formula `0.168*(1-rho)` says seed-to-seed val spread is worth only ~0.0017, and items scoring +0.002 to
+0.003 (distillation, Muon) sit right at it. A candidate that lands at +0.001 is not a result.

Two facts frame the whole list:

- The metric is |y|-weighted Pearson, which is **affine-invariant in the prediction**. Level bias and
  prediction scale are free; only the weighted angle between `p` and `y` is scored, and only rows with
  large |y| carry weight. Every loss-shaping idea here is really an argument about which rows and which
  part of the error budget that angle spends.
- Adversarial AUC 0.98 means the val block is a different distribution, not a noisy sample of the training
  one. So are the rows *within* train that resemble it, and any weighting scheme has to be argued in those
  terms rather than as "regularization".

---

## 1. Train against the metric itself: differentiable |y|-weighted correlation loss

**Mechanism.** With `w_i = |y_i|`, the metric is

```
r_w = sum w (p - mu_w(p)) (y - mu_w(y)) / sqrt( sum w (p - mu_w(p))^2 * sum w (y - mu_w(y))^2 )
```

MSE optimizes the conditional mean under uniform weights and cares about the offset and the scale of `p`,
neither of which `r_w` scores. The existing `max(0.25, 1/(1+since/32))` factor is a hand-built proxy for
the metric's real weighting (rows near an event have large |y|), so it is fitting the metric indirectly and
with a floor that was probably tuned by hand.

**Exact change.** Add a second term computed per batch over both targets' rows:

```
L = huber(p, y, delta=1.0)  +  lam * (1 - r_w)
```

`lam` warmed 0 -> 0.5 over the first 2 epochs (a pure correlation loss has gradient ~1/||p|| and will
diverge from a cold start). Weight the Huber term by the existing since-weight and by `|y|` as well; the
`r_w` term already contains its own weighting, so do not double-weight it. Standardize `p` inside the
loss by its own weighted mean/std (keep the gradient flowing through the denominator) so the term is
numerically stable. Log the per-|y|-decile share of the numerator every 200 steps.

**Why it could beat the list.** Every prior gain is a proxy: the since-weight proxies |y|-weighting,
distillation proxies smoothness, Muon proxies conditioning. This one is the objective, and its gradient has
a zero weighted-mean property that makes the model indifferent to the block-level shift in the target mean,
which is one of the things a 0.98-AUC regime change actually moves. If the proxy is mispriced anywhere
(the 0.25 floor, or `1/(1+since/32)` being the wrong functional form), this recovers it.

**Falsifier.** Sweep `lam` in {0, 0.25, 0.5, 1.0} at fixed seeds. If the best is `lam=0`, or the best gain
over the incumbent Huber is < 0.002, kill it. Secondary: if the top |y| decile already supplies > 70% of
the numerator on the incumbent, the weighting axis is nearly saturated and the remaining headroom is only
the level/scale terms, which are small.

**Cost.** No model change, no extra data pass. ~30 min per arm.

---

## 2. Exploit the rounding: ordinal head over the label support with expected-value readout

**Mechanism.** The label is a *rounded* mean, clipped to +-2. If the rounding grid is coarse (integer
ticks, or a fixed decimal step) then `y` lives on a small support with a large atom at 0 and thin tails.
Regression + MSE fits `E[y|x]` and splits capacity evenly over the support; the |y|-weighted metric only
pays for the tail. A categorical head learns `P(y = c | x)` and the readout `E[y] = sum_b p_b c_b` is both
better calibrated in the tails and explicitly downweights the 0-atom through class weights, while still
using those rows to sharpen the shape of the distribution (unlike hard row downweighting, which throws them
away).

**Exact change.** Replace the 2 regression outputs with `2 * k` logits, `k` = number of observed label
values in [-2, 2] plus two overflow bins; if the distinct-value count is large (check the histogram first),
use `k = 9` fixed bins of width 0.5 and treat it as soft binning. Loss:

```
CE(logits, bin(y), class_weight = 0.25 + |c_b|)  +  0.25 * L1(sum_b p_b c_b, y)
```

The L1 readout term keeps the point estimate sharp; the class weights keep the 0-atom from dominating CE.
Keep the frozen eval identical (it consumes the readout scalar).

**Why it could beat the list.** It attacks the atom-vs-tail imbalance that the |y|-weighted metric creates
and that MSE resolves badly, and it produces a per-row variance `Var_b(c_b)` for free, which is a free
routing signal for idea 8. Nothing on the known-gain list touches the loss's distributional form.

**Falsifier.** First plot the label histogram and the count of distinct values on the training block. If
the label is effectively continuous (thousands of distinct values, no atom at 0 above ~5% of rows), the
discreteness premise is false: kill before training. If it survives that, kill it if the readout
predictions on val are near one-hot everywhere (softmax temperature collapse, no better than regression by
construction) or if val gain < 0.002.

**Cost.** Same architecture size, ~+10% step time. One histogram pass decides whether it is worth running.

---

## 3. Train on the unrounded mean, score on the rounded one

**Mechanism.** Rounding injects approximately uniform(-step/2, +step/2) noise into the target. The model is
asked to predict the quantized version of a quantity it could predict more precisely, and every gradient
carries that quantization noise. Worse, the metric clips at +-2, so extreme |y| rows beyond the clip pay
nothing while still supplying large gradients during training.

**Exact change.** Reconstruct the continuous label `z = mean(mid over next 4 book-price events) - mid_now`
from the mid series (present in the 112 features, or from the label generator). Train on `z` using weights
taken from the *rounded* label (`w = |round(clip(z))|`, since that is what the metric uses). Clip `z` at
+-3 to bound the tail. Keep eval on rounded, +-2-clipped labels, unchanged. Do not add an auxiliary
`L1(p, round(p))` term to "match" the quantized target; that re-injects the noise this change removes.

**Why it could beat the list.** It reduces target-noise variance for free, it makes `z` smoother in `x`
(no step discontinuity at bin boundaries, which matters for a 1M-param GRU sharing capacity across 112
features), and it stops the loss from chasing moves that the clip already discards. This is the cheapest
item on the list: no model change at all.

**Falsifier.** Compute the rounding step and the val residual std on high-|y| rows. Expected gain scales
roughly with `(step^2/12) / Var(z)`. If `std(resid) > 5 * step`, quantization is <5% of the error budget
and the ceiling is under the noise floor, so kill without training. Also kill if the metric's own
weighting changes the picture: check that `|round(z)|` and `|z|` weight deciles agree to within a few
percent, otherwise the weighting swap confounds the result.

**Cost.** One label-reconstruction pass. Zero change to the model or the serving path.

---

## 4. Kill the hand-set since-weight: a per-row noise scale from the event clock

**Mechanism.** `max(0.25, 1/(1+since/32))` assumes label noise is a fixed, monotone, one-parameter function
of the distance to the last price event. Two things are wrong with that: (a) the noise also depends on how
many events actually fall inside the label window (if the 4 events are far apart, the label averages
mid-prices far apart in wall time and is noisier), and (b) the 0.25 floor is a hard cutoff that keeps
weight on rows whose labels are nearly pure noise.

**Exact change.** Add a noise head: `log_sigma = MLP([h_t, log1p(since), log1p(events_in_window),
rolling_abs_ret_32])`, one output per target. Loss:

```
L = sum_i w_i * [ (y_i - p_i)^2 / (2 sigma_i^2) + log sigma_i ],   w_i = |y_i|
```

with `sigma_i` **detached** for the squared-error term for the first epoch (initialize
`sigma_i0 = sqrt(1/weight_i)` so the run starts exactly at the incumbent) and a floor `sigma >= 0.05` to
prevent collapse. Report `Spearman(sigma_hat, |resid|)` on val and `Spearman(sigma_hat, since)`.

**Why it could beat the list.** The since-weight gave +0.003, which is the largest single item known, so
the noise-weighting axis is live. This replaces a one-parameter hand fit with a conditional estimate that
can use event density and volatility, and NLL pays for *irreducible* noise rather than for *distance*, so
it keeps the weight on distant-but-quiet rows that the 0.25 floor currently mistreats in the other
direction.

**Falsifier.** Fit `sigma_hat` on a held-out fold and check its shape: if `Spearman(sigma_hat, since) > 0.9`
and the best single-parameter refit of the current weight (free exponent and floor) recovers the gain, the
hand weight is already near-optimal, kill. Also kill on collapse: `Spearman(sigma_hat, |resid|) < 0.1`, or
`mean(sigma_hat)` moving by >3x with no val change (the model is buying NLL with constant inflation).

**Cost.** +1 small MLP, ~+5% step time. Cheapest non-zero-cost idea after #3.

---

## 5. Supervise the atoms: four single-event increment heads with a composition penalty

**Mechanism.** The label is exactly `1/4 * sum_{k=1..4} r_k` in event time, where `r_k` is the mid change
between the (k-1)-th and k-th next book-price event. Only the sum is supervised today, so a row contributes
one scalar and rows where the 4th event falls outside the sequence contribute nothing usable. The atoms
are individually predictable (the first is nearly contemporaneous) and give four times the gradient signal
per row.

**Exact change.** Add 8 outputs (4 horizons x 2 instruments) predicting `r_k`, built from the mid series.
Loss term:

```
0.3 * mean_k huber(p_k, r_k, delta=1.0)  +  0.1 * | p_4event - mean_k p_k |
```

where `p_4event` is the existing head. The composition penalty is the event-clock constraint: the 4-event
prediction must equal the mean of its own atoms. At serving time you may keep the existing head or replace
it with `mean_k p_k` (that variant is an implicit 4-horizon ensemble at zero extra parameters; test both,
they differ by one line).

**Why it could beat the list.** The aux heads are hand-engineered horizon ensembling, and they recover the
rows near sequence ends that the 4-event label cannot cover (this is *event-clock structure*, not "more
features"). The composition penalty ties the deployed head to the atom heads, so the extra supervision
actually transfers instead of acting as an unrelated regularizer. Note this is not "longer BPTT" or
"depth": the trunk and the serving cost are untouched.

**Falsifier.** Measure `corr(p_1, r_1)` on val for the atom head alone. If 1-event increments are
essentially unpredictable (`corr <= 0.01`), the heads are pure noise and the run is dead. Check also the
coverage argument: if fewer than 2% of rows lose the 4-event label to sequence-end truncation, the coverage
benefit is gone and only the regularization case remains, so require the gain to hold at 3 seeds.

**Cost.** 8 extra scalars out of a 1M-param trunk, negligible step-time. The label build is one pass.

---

## 6. Stateful truncated BPTT with a detached carry

**Mechanism.** Serving is continuous and causal on one CPU core; training chops each 20k-row sequence into
windows with a **cold hidden state at every window start**. So every row in training is conditioned on at
most `W` rows of history, while every row at inference is conditioned on the whole sequence. "Longer BPTT:
no gain" tests the memory-bound version of this. A detached carry tests the mismatch itself at zero extra
memory: gradients still only flow `W` steps, but the *forward* state at row `t` is the true one.

**Exact change.** Iterate windows in order within each 20k sequence, carrying `h` across the backward
boundaries as `h = h.detach()`. Nothing else changes: same window length, same optimizer, same step count.
Second variant, one flag: place window boundaries at rows where the mid price changed (event-aligned), so
that window starts are not systematically event-dense.

**Why it could beat the list.** It is the only cheap fix for a real train/serve distribution gap, and the
gap is largest exactly where the label's clock is densest (clusters of book-price events after a quiet
period), which is where the metric's weight sits. It also interacts with #4: a warm state changes the
predictability of rows early in a window, which is what the since-weight is trying to model with a
row-position proxy.

**Falsifier.** On the current stateless run, plot val error against offset-in-window for offsets 1..256. If
that curve is flat (no excess error at offsets 1-64), the cold start costs nothing and this is dead. If it
is not flat, the gain must appear at the early offsets specifically; if it appears only as a uniform
+0.001 it is probably noise and should be rejected against the 0.002 floor.

**Cost.** Ten lines in the data loop, no memory increase. Run this one early because its diagnostic (the
per-offset error curve) is useful for everything else.

---

## 7. Density-ratio reweighting with shrinkage, clipping, and an ESS report

**Mechanism.** With AUC 0.98 the val block is a distinct region of feature space, and uniform weighting of
training rows optimizes the wrong risk. The since-weight is a *predictability* prior; this is a
*relevance* prior, and they compose multiplicatively.

**Exact change.** Fit a shallow LightGBM (or logistic on standardized features plus 3-scale rolling
means/stds of them) to predict train-vs-val membership, 5-fold **out-of-fold** on train rows so no row is
scored by a model that saw it. Then

```
w_i = clip( ((p_i/(1-p_i))^tau) / median , 0.1, 10 ),  tau = 0.5
```

multiplied into the existing since-weight and into the |y| weight. Report effective sample size
`ESS = (sum w)^2 / sum w^2`, the top-5 split importances, and the weight distribution. `tau = 0.5` is
deliberate: with near-perfect separability the raw ratio is degenerate, and shrinkage is what keeps the
variance finite.

**Why it could beat the list.** It is the standard correction for the exact condition that defines this
task's validation, and it is orthogonal to every item on the known-gain list. If it fails, the importances
are still worth having: they usually reveal that the shift is concentrated in one or two features, in which
case a per-block feature normalization (or a rank-transform of those two features) is the cheaper fix and
should replace this idea.

**Falsifier.** Build two additional held-out shifted blocks by the same block-splitting rule. If the
weighted model loses on 2 of 2 alternates, or if `ESS < 0.2 * N` and the sign of the delta flips between
`tau = 0.25` and `0.5`, kill. If the top split importance explains >50% of the classifier's gain, the shift
is low-dimensional: replace reweighting with normalization of that feature and report that instead.

**Cost.** One LightGBM fit on CPU, minutes. No change to the GRU.

---

## 8. Event-phase specialists, and measure their residual correlation instead of their individual scores

**Mechanism.** Seed ensembles correlate 0.99, so the blend gain is `0.168*(1-rho)` and the only lever left
is **structural** error decorrelation, not more samples of the same error. The since-weight gain is direct
evidence that the conditional distribution of the label changes with phase since the last event: rows at
phase 0-8 are a different problem from rows at phase 32+. A single 1M-param GRU fits all phases with one
set of weights, so its systematic error is shared across the whole population. That shared systematic
component is exactly what makes `rho = 0.99` across seeds.

**Exact change.** Two or three students at hidden 160-192 (keep total params within ~1.2x of the current
model so serving stays one core). Each is trained only on rows whose `since` lies in its band
(`[0,8) / [8,32) / [32,inf)`), with the |y|-weighted Huber loss, ~3 epochs. At inference, gate by phase:
hard-select, or blend with softmax weights over `-log1p(since)` distance to each band. Report the pairwise
val residual correlation of the three models, and their individual val scores.

**Why it could beat the list.** The expected gain is bounded by the blend formula, not by any single model:
pairwise `rho = 0.95` gives ~0.008, three times the best known item. Phase splitting is the cheapest
structural axis available because it needs no extra labels and is directly suggested by the one weighting
trick that already works. If the experts' `rho` stays above 0.98, the result is still informative: it says
the error is dominated by something phase-independent (likely the feature-level shift), which redirects
effort to #7.

**Falsifier.** Compute `rho` between experts on val before looking at the blend. If `min rho > 0.98`, kill.
Separately: plot the current global model's val error by since decile. If it is flat, the premise that
phases are different problems is false, and the whole item is dead regardless of the blend number.

**Cost.** ~1.2x training compute of one model, split across phases (each expert sees ~1/3 of the rows, so
wall-clock is comparable). Serving is the real constraint: check the 160-hidden CPU latency before
committing.

---

## Suggested order and composition

Run the diagnostics first, they are nearly free and they kill or confirm half the list: label histogram and
distinct-value count (#2, #3), per-offset error curve (#6), val error by since decile (#8), top split
importances from the domain classifier (#7).

Then, in order of expected value per hour: **#3** (zero model change) and **#6** (ten lines) in parallel;
**#1** as the single most likely structural win because it removes three proxies at once; **#8** as the
highest ceiling but the only item with a serving-cost risk; **#4** as a refinement of the already-working
weighting; **#5**, **#2**, **#7** last.

Compatibility, if two candidates both clear 0.002: #1 + #3 (same loss, better target), #4 + #7 (two
independent weights multiplied into one), #5 + #6 (aux heads benefit most from a warm state), #8 with any
of them applied per expert. Do not stack #1 with #4 before each is measured alone, since both change the
effective per-row weighting and the debug will be ambiguous.
