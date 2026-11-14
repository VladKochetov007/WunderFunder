# Six ways to get more out of the distillation signal

The incumbent: cold 1M GRU, target `0.5*y + 0.5*t`, |y|-weighted MSE with event-age row weights,
4 epochs, val 0.6905 against teacher 0.6914 and labels-only 0.6867.

Three things about this setup shape everything below.

**The gap you are chasing (0.0009) is smaller than the noise you measure with.** Seeds correlate 0.99, so
`0.168*(1-rho)` puts seed-to-seed spread at ~0.0017. Reporting "0.6905 vs 0.6914" at four decimals is not
a result. Score the two models on the *same* val rows and bootstrap the paired per-row
difference over the val block's sequences (1000 resamples, resample sequences not rows, since rows within a
sequence are dependent). Every idea below is judged on that paired delta, not on the mean of seeds. If a
change cannot clear its own paired 95% interval, it does not exist.

**Teacher predictions exist only on training sequences, and there they stay.** So the teacher cannot be a
serving-time input, and cannot be used to fit anything on the val block. Any β you calibrate, you calibrate
on a held-out slice of *train* sequences (hold out ~500 sequences and keep them out of every fit that
produces the calibration).
This rules out the tempting "just add the teacher at inference" family entirely and is why the ideas below
are about targets, rows, and schedules.

**Why the 0.5 mix doesn't overfit and labels-only does.** With `beta = 0.5` the target retains roughly half
the label's high-frequency component, and that half is enough to keep the last epoch from chasing noise,
while the teacher half supplies the low-frequency function. This is a statement about *which frequency band
the target exposes at which point in training*, and ideas 1, 2, and 6 are three different ways of exploiting
it.

One measurement splits the whole space before you spend a run: take `r = y - t` on train rows, cross-fit a
small GBM on the 112 features to predict `r`, and report the out-of-fold weighted correlation
`corr(r_hat, r)`.

- If it is ~0, the teacher already captures everything the features contain and the only headroom is
  **target-noise reduction** (ideas 1, 2, 6). "Student beats teacher" is then out of reach and the goal is
  to land within noise of 0.6914 much more reliably.
- If it is clearly positive, the teacher has feature-explainable blind spots, and **idea 3** (and its
  by-product, idea 4) is where the only route above the teacher lives.

---

## 1. Per-row mixing weight, calibrated on measured noise rather than set to 0.5

**Mechanism.** A global 0.5 asserts that label and teacher deserve equal weight on every row. They do not.
Label noise here is close to additive and row-independent (a rounded mean over a jittered event window plus
quantization), so its variance is roughly flat in |y|. Teacher error is a function error, so it grows with
signal magnitude and shrinks toward zero near y=0. The consequence: **the optimal β rises with |y|** and,
because the label's window drifts in event time, **falls with event age**. The 0.5 mix currently dilutes the
label on exactly the rows the |y|-weighted metric cares about and dilutes the teacher on the rows where the
label is pure noise.

**Exact change.**

```
beta_i = a_i^2 / (a_i^2 + c_i^2),      a_i = |y_i|
c_i    = c0 * sqrt(1 / age_weight_i)
```

Pick `c0` numerically so that the |y|-weighted mean of `beta_i` equals 0.5, which keeps the total amount of
teacher in the target constant and makes this a pure redistribution. Clip to [0.15, 0.85]. Target becomes
`beta_i*y_i + (1-beta_i)*t_i`; the loss weights stay untouched (this must not be confounded with the |y| and
age weights, which are already tuned).

**Why it can beat 0.5.** The one gain from mixing (no last-epoch overfit) is preserved by construction, while
the mixing is spent where it is worth something: teacher on the noise-dominated small-|y| rows, label on the
large-|y| rows that carry the metric. It also composes with everything else here.

**Falsifier, and do this before training anything.** Plot `Var(y - t)` against |y| decile and against event
age decile on held-out train rows. The premise requires `Var(y-t)/Var(y)` to fall with |y| and rise with age.
If `Var(y-t)` is flat or rising in |y| fast enough that the ratio is constant, the additive-noise model is
wrong and β should not be moved by |y|: kill. Confirmation run: three decile-bucketed students (bottom 40%,
middle 40%, top 20% of |y|) at β ∈ {0.3, 0.5, 0.7}, 9 runs, and check that the optimal β is monotone in the
decile. If the per-decile optimum is flat, the whole per-row axis is saturated.

**Cost.** Same model, same steps, one label-preprocessing pass.

---

## 2. Average the three teachers, weighted by error variance, and measure error covariance first

**Mechanism.** The three teachers at 0.6914 / 0.6878 / 0.6874 have prediction correlation ~0.99, which
sounds like averaging gains nothing. But the 0.99 is on *predictions*, and predictions correlate because
they all track the same signal. What matters for a distillation target is the correlation of *errors*,
`corr(y - t_j, y - t_k)`. If errors are considerably less correlated than predictions, the mean teacher is a
strictly quieter target, and target noise is what the student integrates: with `n` teachers,

```
Var(target noise) = (1/n^2) * [ sum_k sigma_k^2 + 2 * sum_{j<k} rho_jk sigma_j sigma_k ]
```

At `rho_jk = 0.99` this is ~nothing; at `rho_jk = 0.7` it is a ~35% cut, which is comparable to the entire
distillation gain. Also, the 0.6874 teacher is the one a val-score-weighted average would mostly ignore, and
it may be the one with decorrelated errors, which is precisely the one worth keeping.

**Exact change.** Compute `sigma_k` (|y|-weighted std of `y - t_k` on held-out train rows) and `rho_jk` on
the same residuals. Set `w_k = (1/sigma_k^2) / sum_j (1/sigma_j^2)`, `t_bar = sum_k w_k t_k`, and use
`0.5*y + 0.5*t_bar` with the β from idea 1 replacing the 0.5s if that one cleared its falsifier. Keep the
val scores out of it. Store all three teacher outputs on the train sequences once; they are pure
preprocessing for every idea in this document.

**Why it can beat the single-teacher run.** It reduces target noise without touching the model, the loss, or
the schedule. And if the error covariance turns out to be low while the prediction correlation is high, that
single measurement also explains a chunk of the remaining 0.0009: the student is being asked to imitate a
noisier function than the ensemble mean.

**Falsifier.** If `min rho_jk > 0.97` on residuals and `sigma_3/sigma_1 > 0.9`, the mean is
indistinguishable from teacher 1 and this is dead: skip the run. If the mean is used and the paired delta is
inside noise, also check that `|t_bar - t_1|` is not concentrated on rows where `|y|` is tiny (in which case
the change is real but metric-irrelevant).

**Cost.** One offline pass of three teacher models over train. Zero training-time change.

---

## 3. Learn where the label beats the teacher: a cross-fitted residual gate

**Mechanism.** To exceed 0.6914 the student must beat the teacher on rows where the teacher is *wrong*, not
just reproduce it. Those rows are identifiable from data you already have: if `r = y - t` is predictable
from the 112 features by a model that never saw the teacher, then the teacher is missing feature-level
structure and the label is the better target there. If `r` is not predictable, the row's disagreement is
label noise and the teacher is the better target. This replaces the guess "teacher and label disagree, so
believe X" with a measurement, and it is cross-fitted so the gate cannot memorize train rows.

**Exact change.**

1. `t_bar` from idea 2, `r = y - t_bar` on train rows, |y|-weighted.
2. 5-fold cross-fit a small LightGBM (`num_leaves=31`, 200 trees, lr 0.05) on the 112 features to predict
   `r`, giving out-of-fold `r_hat_i` and `g_i = |r_hat_i| / std(r_hat)` (also report OOF weighted
   `corr(r_hat, r)` overall and per |y| decile).
3. Gate: `beta_i = 0.5 + 0.25 * tanh(g_i - 0.5) - 0.20 * tanh(|r_i| / p90(|r|) - 1)` clipped to [0.1, 0.8].
   Read: raise label trust where the residual is feature-explainable (teacher blind spot), lower it where the
   residual is a large and unexplainable spike (label noise).
4. Train as usual with `beta_i*y_i + (1-beta_i)*t_bar_i`.

**Why it can beat the list.** Ideas 1, 2 and 6 only reduce target noise, which cannot lift you past the
teacher by construction: they converge to the teacher function. This is the only item that changes *what
function the student is asked to fit*. It also writes an explicit answer to "who is right when they
disagree", and its diagnostic (`corr(r_hat, r)` per |y| decile) tells you whether the top |y| decile, which
carries the metric, is where the teacher's blind spots are concentrated.

**Falsifier.** If OOF `corr(r_hat, r) < 0.02`, the teacher has no feature-explainable error and the entire
premise fails: kill, and stop looking for teacher-beating ideas. If it is high (>0.15) but the gated run lands
inside noise, then the student cannot represent the correction the GBM found (capacity, not information) and
the honest next step is a wider student, not more target engineering. Guard against the gate degenerating:
report `std(beta_i)`; if the tanh terms cancel and it is ~0.5 everywhere, dead.

Do not add `r_hat` to the target baseline as a serving-time term. That is an ensemble, not distillation,
and it changes the serving budget.

**Cost.** One LightGBM cross-fit on CPU, minutes; one extra training run.

---

## 4. A second distillation round with a three-way agreement vote

**Mechanism.** After round 1 you own two function estimates (the teacher and the student) that are *not*
nested: a 1M GRU is not an approximation of a wider GRU in any useful sense. On a given row, three things can
happen: label, teacher and student all disagree (genuinely hard, and the label is the only independent
arbiter, so believe the label), or teacher and student agree while the label is far off (two independently
trained models landing on the same value is strong evidence the label is corrupt, so believe the teacher),
or the teacher and student split (informative about which one has the local defect, resolved by the label).
This is co-teaching's filter with the vote cast by models of different families and widths instead of two
seeds of the same net, which matters because same-family seeds here agree 0.99.

**Exact change.** From round-1 checkpoints (all 3 seeds, and their EMA if it exists; use the median
prediction `s` to avoid one bad seed dominating), compute per row on the train block
`a_i = |y - t_bar|`, `b_i = |s - t_bar|`, `c_i = |y - s|`, each normalized by its own p90.

```
both_agree_label_out  (b small, a large)  -> beta = 0.10
all_disagree          (a, b, c all large) -> beta = 0.70
otherwise                                 -> beta = 0.50
```

Soft version: `beta_i = 0.5 - 0.4*tanh(2*b_i_n - 1)*tanh(2*a_i_n - 1) + 0.2*tanh(a_i_n + b_i_n + c_i_n - 2.4)`,
clipped [0.05, 0.85]. Round 2 is a fresh cold student, 4 epochs, same seeds, target `beta*y + (1-beta)*t_bar`.

**Why it can beat the list.** The `both_agree_label_out` cell is a label-noise detector that no single model
can provide, and label noise is the binding constraint on the late epochs (the observed last-epoch overfit).
The `all_disagree` cell is the only place where the label's unique information can enter without also letting
noise in. It also tells you the size of the irreducible label-noise floor on this block, which bounds how far
any target engineering can go.

**Falsifier.** Report the |y| distribution of the `both_agree_label_out` set. If more than ~30% of its mass
is in the top |y| decile, the vote is flagging the rows that dominate the metric and trusting the teacher
there could cost more than it saves: shrink the β shift to 0.3 and re-check, or kill. Separately, if
`corr(s - t_bar, y - t_bar) > 0.9` the student is a near-copy of the teacher, the vote is a duplicate, and
round 2 is just round 1 with extra steps.

**Cost.** One extra training run plus one inference sweep of the round-1 student over train.

---

## 5. Teacher on input-transported rows: supervision where the regime shift is

**Mechanism.** The val block is a different distribution (AUC 0.98) and the teacher exists only on train, so
the student currently gets **zero** supervision anywhere near the val feature distribution. Distillation is
the one mechanism that can manufacture targets there without labels: transport train rows into the val
feature geometry, ask the teacher, and add those pairs as extra training data. This is not the closed
density-ratio item. The row weights on real rows stay exactly as they are; what changes is that new,
label-free rows appear.

**Exact change.**

1. Estimate the train→val shift cheaply: for each feature, `(mu_val - mu_train)/sigma_train`, and a
   rank-transport map `x -> F_val^{-1}(F_train(x))` per feature (monotone, preserves within-row dependence
   better than affine).
2. Pick the top-K features by |standardized mean diff| (K ≈ 10-20) and transport only those, leaving the rest
   identical. Transporting all 112 destroys the row structure.
3. Run the 3-teacher mean on the transported rows, obtaining `t_tilde`. These rows carry no label.
4. Loss on transported rows: `0.25 * (p - t_tilde)^2` (|y| weighting is meaningless there, so weight by
   `age_weight` only; age is defined from the row's own event clock and survives transport).
5. Keep the real-row term exactly as in the incumbent.

**Why it can beat the list.** Every other item optimizes the student on the training distribution and hopes
the 0.98-AUC shift is benign. This is the only idea that puts target-bearing rows into the shifted region.
It also uses the fact that the teacher has *never* been validated on transported inputs; whatever it does
there is a smooth extension of a function that scores 0.6914, which is a better prior than the student's own
extrapolation, which is what the current cold student does on shifted rows.

**Falsifier, and run this first.** Build a proxy shift inside train: choose 500 sequences whose
adversarial separability from the rest most resembles the real split, hold them out, and repeat the whole
procedure with the teacher refit on the remaining 9500. The proxy block has labels, so you get a real number
for the transported-distillation idea in an hour, without risking the actual val block. Kill if the proxy
delta is inside noise. Second guard: compare the distribution of `t_tilde` with `t` on the same rows. If
`mean|t_tilde|` drops by more than ~40% or `std(t_tilde)` collapses, the teacher is out of distribution and
its transported outputs are a shrinkage prior, not information; in that case reduce the transported weight
to 0.05 or kill.

**Cost.** One teacher pass over transported rows plus 1.5x train steps (the extra rows are cheap: no labels,
no backward through anything else).

---

## 6. Put the teacher where the overfit happens: a β schedule over the 4-6 epoch budget

**Mechanism.** The fact that the 0.5 mix does not overfit in the last epoch while labels-only does is
evidence that the teacher term suppresses late high-frequency fitting. Under the usual frequency view, early
epochs fit the low-frequency, generalizing component of the label and late epochs fit high-frequency noise.
So the correct schedule is *label-heavy early, teacher-heavy late*, which is the opposite of the intuition
that you should start from the clean signal. Early on you want the label's corrections, because the teacher's
blind spots (idea 3) are bias you should fix while the model is still flexible; late you want the teacher,
because that is the epoch where the label's noise is being fit.

**Exact change.**

```
4 epochs:  beta = [0.30, 0.45, 0.60, 0.75]
6 epochs:  beta = [0.30, 0.45, 0.60, 0.75, 0.85, 0.90]   with cosine LR -> 0
```

Same `t_bar`, same weights, same seeds. This is two lines in the training loop, and the 6-epoch arm directly
probes the "rapid overfit after 4-6 epochs" boundary from the Muon result: if the schedule buys epochs, the
saving is larger than any target tweak here.

**Why it can beat 0.5.** It keeps 100% of the mix's anti-overfit benefit while recovering the label budget
that a constant 0.5 wastes in the first two epochs, which is where the teacher's bias is still correctable.
It also composes with ideas 1-4, all of which supply a `beta_i`; the schedule multiplies it as
`beta_i * m_e` with `m_e = [0.6, 0.9, 1.2, 1.5]` clipped.

**Falsifier.** Using the constant-β runs from idea 1, check whether the ranking of β=0.3 vs β=0.75 *changes*
between epoch 2 and epoch 4. If β=0.75 is already better at epoch 2, or β=0.3 still better at epoch 4, the
per-epoch optimum is not moving in the assumed direction and the schedule is dead: use constant β. Second
check: plot the val curve at every epoch for both arms. A schedule that helps should help at epoch 4
specifically; a flat final epoch means it only re-shaped the trajectory.

**Cost.** Zero. Two lines, same compute.

---

## Order

Run in this order, and stop early where a falsifier fires.

1. **Offline, no training:** three-teacher outputs stored (idea 2 setup), residual correlation matrix (2),
   the `corr(r_hat, r)` probe (3), `Var(y-t)` vs |y| and vs age (1). These four diagnostics cost minutes and
   together decide whether you are in a "reduce target noise" world or a "teacher has blind spots" world.
2. **Idea 6**, because it is two lines and it reuses the β sweep runs from idea 1's falsifier.
3. **Idea 2** if the residual correlations are below 0.97; it is pure preprocessing.
4. **Idea 1**, calibrated with the per-decile optima you already measured.
5. **Idea 4**, which needs a round-1 student, so it comes after one of the above has produced a better one.
6. **Idea 5** last, and only through the proxy-shift experiment.

Report every arm with the paired per-sequence bootstrap delta against the incumbent and against teacher 1 on
the same rows. With a 0.0009 target and a ~0.0017 seed spread, an unpaired comparison cannot distinguish
success from noise, and the last thing this problem needs is another +0.001 that does not replicate.
