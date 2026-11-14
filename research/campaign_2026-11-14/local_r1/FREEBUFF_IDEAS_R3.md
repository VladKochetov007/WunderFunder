# Combinations as a construction kit

The 2x2, one seed, 4 epochs:

|                  | labels only | + teacher (0.5 mix) |
|------------------|-------------|---------------------|
| no age weight    | 0.6859      | 0.6881              |
| age weight       | 0.6867      | 0.6905              |

Interaction `I = 0.6905 - 0.6881 - 0.6867 + 0.6859 = +0.0016`. The asymmetry is the interesting part:
the age weight is worth **+0.0008 alone** (below the 0.001 seed sd, i.e. nothing) and **+0.0024 with the
teacher**. The mix is worth +0.0022 alone and +0.0038 with the age weight. Both together are +0.0046, which
is more than the sum of the two standalone numbers (0.0030). Any mechanism has to explain why
one of the two knobs is nearly inert on its own. "Both reduce variance" does not explain it, because two
variance reductions on the same target are *substitutes*, not complements:

```
target m = 0.5*y + 0.5*t,  error m - f* = 0.5*(e_y + e_t)
Var(m - f*) = 0.25*(Var(e_y) + Var(e_t) + 2*Cov(e_y, e_t))
```

Mixing quarters the label-noise term. The excess variance of the old rows that the age weight targets is
mostly a *label* quantity, so mixing shrinks it by 4x as well — under this algebra, adding the mix should
make the age weight *less* valuable, and the measured sign is the opposite. So either the interaction is
not a variance story, or one of the two knobs is doing something other than variance reduction. That is the
space the three explanations below divide up.

**Before any of it: the interaction has no error bar.** One seed per cell means the four scores are four
points from one deterministic function of the seed. If the four cells shared a seed (same init, same data
order), `I` is exactly invariant to the common seed effect and the four `+1,-1,-1,+1` weights cancel it, so
`I` can be much more precise than the marginals — but its variability *across seeds* is still unmeasured,
and a paired bootstrap over val sequences covers val sampling only. If the cells did not share a seed,
`Var(I) = 4*sigma_seed^2`, i.e. `+0.0016` against `0.001` is not a result at all. The screen in section 3
spends its first two runs buying that ruler, and every mechanism below is conditional on `I` surviving it.

---

## 1. Three candidate mechanisms, and the diagnostics that separate them

Each diagnostic is zero-training: it uses the four saved checkpoints' predictions or the data, not a new
40-minute run. They are built to give *different* numbers, so that one run of the diagnostics tells you
which story you are in.

### E1 — The teacher is the memory of the rows the age weight deletes

**Mechanism.** `w = max(0.25, 1/(1+since/32))` is information deletion. On labels only, deleting old rows
costs you the function information those rows carry, and that cost is why the age weight alone is +0.0008.
Under distillation the cost is prepaid: `t` is a function fitted on all rows, so the information in the
downweighted rows has already been absorbed into a smooth global estimate. The student can then downweight
aggressively without losing the global shape, because it is no longer estimating that shape from the
surviving rows alone. Super-additivity here is a *transfer* effect: the age weight stops being an
estimator-variance trade and becomes a pure re-weighting of a target that already knows what it threw away.

**Zero-training diagnostic.** Per `since` decile, on train rows, compute `Var(t)` and the teacher's *train*
residual `Var(t - y)`, next to its val residual on the same decile. A teacher trained with the same age
weight has already collapsed toward the per-decile mean on old rows, and shows it as a small `Var(t | old)`
together with a large train residual there (it downweighted those rows, so it did not fit them). The
signature for E1 is `Var(t | oldest decile) / Var(t | freshest decile) > 0.6` — the teacher retains
substantial conditional variance on exactly the rows the student deletes, i.e. it really is carrying
information the student would otherwise lose. If that ratio is below 0.4 the teacher has forgotten the same
rows, E1 is dead, and the pair's complementarity is not a transfer effect.

**Prediction if true.** The interaction grows as the age-weight floor drops (0.25 → 0.1) and as the mix
weight on the teacher rises. It should also be visible as a positive interaction between the age weight and
*any* strong prior on the function (EMA-teacher, a wider student), not specifically the teacher target.

### E2 — Two noise components, and the |y| weight sitting on the one nobody removes

**Mechanism.** Decompose the label's deviation from the true conditional mean into
(i) **horizon noise**, which grows with `since` because the 4-event window is further away and consecutive
labels overlap more heavily, and (ii) **flat noise** from rounding/quantization, roughly independent of
`since` and independent across rows. The age weight reduces (i) by downweighting the rows where it is large;
the mix replaces both components with a quarter of their variance plus a quarter of the teacher's own error.
Neither alone removes both, and whichever component survives is the one that binds the 4-epoch fit — that is the "binds" part that turns additivity into super-additivity, and it is where the
metric's `|y|` weight matters:

```
L = sum_i |y_i| * (p_i - m_i)^2
m_i - y_i = 0.5*(t_i - y_i)
```

The mix injects a *bias* of `0.5*(t - y)` into the regression, weighted by `|y|`. Because the teacher is a
smoother, shrunken function, `|t - y|` grows with `|y|`, so the mix's worst rows are exactly the rows the
metric pays for. The age weight cannot fix them: `|t - y|` is a property of teacher disagreement, not of
`since`. That leaves a specific residue — large-`|y|`, teacher-disagreement rows carrying maximum loss
weight — which neither knob touches and which therefore sets the ceiling of the pair at 0.6905, 0.0009
under the teacher. Any third block that removes *that* residue is super-additive with the pair by
construction, because the residue only exists once the mix is on.

**Zero-training diagnostic.** Two numbers. (a) Regress `|y - t|` on `since` and `|y|` on train rows with a
small GBM; report weighted `R^2`. If `since` alone explains most of it, the age weight is already handling
disagreement and a robustness-style block will be a *substitute* (negative interaction). If it explains
almost none, disagreement is an independent component and the third block is orthogonal. (b) The share of
`|y|`-weighted loss mass in the top 0.5% of `|y - t|` rows. Below ~5%, the residue is too small to pay for
any block; above ~15%, it is the largest single term left in the objective.

**Prediction if true.** The interaction mass sits in the large-`|y|` × mixed-disagreement bins, and it is
absent in small-`|y|` bins even where the age weight fires hardest.

### E3 — The 4-epoch budget is the binding constraint, and each knob buys a different epoch

**Mechanism.** The one behavioural fact we have about overfitting: labels-only overfits in the last epoch,
the 0.5-mix run does not. Add that Muon overfits after 4-6 epochs, i.e. nothing about this problem is
converged at the 4-epoch checkpoint — the val score is where a *trajectory* stopped, not where a function
converged. On that reading, the age weight slows the fit by removing gradient mass (it changes how much of
each epoch is spent on the old-row directions), and the mix slows it by removing high-frequency target
content. Each alone still leaves the other bottleneck: age-weight-only still fits fresh-row quantization
noise, mix-only still spends steps on horizon noise. Together they survive long enough for the fourth epoch
to be productive, and the interaction is the *unspent* overfit that neither could spend alone.

**Zero-training diagnostic.** The four runs' per-epoch val curves. Compute `I_e` for `e = 1..4`. The
signature is `I_1 ≈ I_2 ≈ 0` and `I_4 > 0`, with the combined arm's per-epoch gain at epoch 4 exceeding the
sum of the single arms' epoch-4 gains. If `I_e` is flat across epochs, the interaction is in the *function*,
not in the trajectory, and E3 is dead. A second, nearly-free check: the val-loss curvature at epoch 3→4 for
each cell. E3 predicts the combined cell is still descending while both singles have turned.

**Prediction if true.** Extending to 6-8 epochs shrinks the interaction (both arms eventually arrive), and
the pair is a *scheduling* trick rather than a target trick. That is a big deal for how much of the
remaining budget goes to schedules.

### Sign bookkeeping (one line, worth doing because it is free)

The four checkpoints give four residual vectors on the same val rows. Take the per-row squared-error
changes `d_age = e_age^2 - e_base^2` and `d_mix = e_mix^2 - e_base^2`, weight by `|y|`, and report their
correlation. Positive correlation means both knobs are repairing the same rows — substitutive, and the 2x2
should not have shown +0.0016. Near zero means disjoint repairs — additive. Negative means they repair
different rows and one of them *worsens* rows the other fixes, which is the only pattern that first-order
predicts super-additivity from. This is a first-order probe (the combined cell is retrained, not summed),
so treat it as a sign, not a magnitude.

---

## 2. Six blocks to stack on the pair

Rule applied throughout: a block is only interesting if it acts on a part of the pipeline the pair does not
already touch — the pair already owns the *target value* and the *row scale*. So everything below is input,
state, optimizer, sampling, readout, or loss geometry. Each is tagged with the mechanism it exploits and
the sign of interaction it predicts.

### B1 — Event-gated second layer (state)

**Mechanism.** The age weight says "this row carries little new information." That is an assumption about
the *conditional mean*: between two book-price events the mid is literally constant, and the label is the
mean over the next four events, so the row's target can only change when an event arrives and when the
window slides. A GRU that updates the whole state at every row is free to fit within-interval book noise.
Freeze the *second* layer between events and the model's function becomes piecewise-constant on exactly the
intervals the label averages over. The pair makes that safe: the mix supplies a smooth function on the
frozen stretches, and the age weight decides how much those stretches count. Labels-only, the freeze would
be fighting real within-interval drift; with the mix it is not.

**Exact change.**

```
e_t   = 1 if mid_t != mid_{t-1} else 0     # the label generator's clock is already known; no inference needed
g_t   = sigmoid(a + b*e_t + c*log1p(since_t)),   a=0, b=4, c=0   # ~0.98 on events, ~0.02 off
h2_t  = (1 - g_t)*h2_{t-1} + g_t*GRU2(x_t, h1_t, h2_{t-1})
```

Layer 1 keeps its per-row update, so book features still enter every row. Serving gets cheaper on
non-event rows, not more expensive.

**Free diagnostic before running it.** On the incumbent checkpoint, measure `|p_t - p_{t-1}|` across row
pairs with `e_t = 0`, as a fraction of `|p_t - p_{t-1}|` on event pairs. If the model already barely moves
between events (ratio < 0.1), the gate is a no-op and the block is dead before training. Second: fit the
within-interval label change from book features; weighted `R^2` near zero means freezing costs nothing.

**Falsifier.** Report mean `g_t` over a run: if it drifts to 1 everywhere the block has been learned away
and the result is uninformative. Kill if the paired delta over the pair is under 0.001, or if it is positive
but concentrated entirely on event rows (where the gate is open and the block does nothing) — that would
mean the gain is a regularizer, not the mechanism.

### B2 — Muon on the recurrent matrices, with the 6-epoch arm (optimizer)

**Mechanism.** This is the one block with a measured standalone gain (+0.003) and a measured failure mode
(overfit after 4-6 epochs). Orthogonalizing the update equalizes the singular-value spectrum, i.e. it
amplifies the low-variance directions of the recurrent weights — which, on a noisy label, are the
directions that fit noise. On a target whose variance has been quartered and whose rows have been reweighted,
those same directions are signal. So Muon's known defect and the pair's known benefit are the same
quantity, which is the cleanest available reason to expect interaction. E3 in one sentence: Muon needs
epochs, the pair buys epochs.

**Exact change.** Muon (5-step Newton-Schulz) on the six recurrent matrices only; AdamW unchanged on
embeddings, embeddings-to-hidden, and readout; LR 1e-3, momentum 0.95, Nesterov, as already measured. Run
both 4 and 6 epochs with cosine decay to 0. Age weight and β = 0.5 fixed.

**Free diagnostic.** From the existing Muon and AdamW checkpoints, split per-row weighted error by `|y|`
quartile and `since` quartile. The mechanism predicts Muon's advantage concentrated in the *high-|y|* bins
(the directions where independent label information lives). A flat advantage means the orthogonalization
story is wrong and the sole reason to stack it disappears.

**Falsifier.** Measure the epoch at which val peaks for Muon+pair vs Muon alone. If the peak does not move
at least one epoch later, the "clean target feeds Muon" mechanism is false; the 6-epoch arm then has no
reason to exist. Kill if the 3-way delta over the pair is under 0.001.

### B3 — Event-clock phase inputs and within-sequence feature normalization (input)

**Mechanism.** The age weight is an output-side scalar guess at a quantity the model may not be able to
represent: where in the event cycle this row sits, how many events remain inside the label's 4-event window,
how far the current mid has drifted from the last event mid. Feeding those directly converts a
multiplicative loss scale into an input, so the network can learn *phase-conditional* functions instead of
averaging over phases and having the weight approximate the average. The mix becomes more useful in the same
stroke, because a smooth target is easiest to fit conditionally. This block is deliberately allowed to come
out **negative**: if phase is already recoverable from the 112 features, adding it is a substitute for the
age weight, and a negative interaction is the cleanest confirmation that the pair is doing phase work.

**Exact change.** Six extra channels per row per instrument: `log1p(since)`, `1/(1+since)`, `log1p(rows to
next event)`, `log1p(events in the trailing 32 rows)`, `mid_t - mid_at_last_event`, and the sign of the last
event. Second variant, one flag: replace global z-scoring of the 112 features with per-sequence
median/MAD scaling, which matters if the shift is a volatility shift.

**Free diagnostic.** GBM/linear probe predicting `since` from the 112 features on train rows, `R^2` out of
fold. Above 0.98: the information is already in the inputs and the phase channels are near-redundant — skip.
Below 0.9: there is a real information gap and this is likely the cheapest block on the list.

**Falsifier.** Kill if the paired delta is under 0.001. Then the informative arm: run B3 with the age weight
off. If its gain is *larger* without the age weight, the two are substitutes and it should not be stacked —
it should replace, and the pair's +0.0024 age term should be re-derived with phase inputs in place.

### B4 — Event-density-matched sequence sampling (sampling)

**Mechanism.** The age weight defines an effective training measure `w(since) * |y|`. That measure only
lands where intended if the batches' `since` distribution matches the distribution you will be scored on.
With adversarial AUC 0.98 the val block is a different region, and if its event density is different, the
same `w` is applied at a different operating point — the model is being trained for the wrong mixture of
event regimes, and the metric pools over all of them. Matching the density puts the age weight at the val
operating point, and it raises the share of dense-event sequences, where the 4-event label has a short
horizon, the mix is most accurate, and the pair's benefit is largest.

**Exact change.** Per sequence, `d_s = events / rows` and `m_s = mean(since)`. Bin into deciles, and sample
sequences with weight proportional to `val_hist(bin) / train_hist(bin)`, clipped to `[0.2, 5]`, multiplied
into the existing sampler. Report `ESS = (sum w)^2 / sum w^2` per epoch. Val histograms come from the val
*features* only; no labels touched.

**Free diagnostic.** KS distance between train and val on `d_s` and on `m_s`, plus the adversarial AUC of a
classifier using only `(d_s, m_s)`. If both KS < 0.05 and that AUC is ~0.5, density is not the shifted axis
and the block is dead before it starts. If the AUC is high (>0.7), the mismatch is real and this is the
cheapest correction available.

**Falsifier.** Kill if `ESS < 0.3N` (the sampler is now training on a handful of sequences), or if the delta
sign flips between clip 5 and clip 10, or if the sampled batches' mean `since` has not moved.

### B5 — Per-phase affine readout calibration (readout)

**Mechanism.** The metric is weighted Pearson, so it is exactly invariant to `p -> a*p + b`. A *global*
affine recalibration therefore cannot change the score; only a recalibration that varies by subgroup can.
The age weight trains the function under a reweighted measure, which shifts the conditional mean differently
in each `since` band, and the mix averages two functions with different per-band slopes. So the pair
plausibly introduces a phase-varying slope distortion — a thing that costs score, is invisible to any
global check, and is fixable with a handful of parameters.

**Exact change.** `p = (1 + A_phi) * (W_r h + b_r) + b_phi`, `phi` = 3 `since` bands × 2 instruments,
`A` and `b_phi` initialized at 0 with weight decay 1e-2 pulling them back. Cheaper variant that needs no
retraining and gives a ceiling: after training, fit `(A_phi, b_phi)` in closed form by `|y|`-weighted least
squares on held-out **train** sequences, then evaluate on val.

**Free diagnostic, and this one can kill or promote the block without a run.** Take the four existing
checkpoints, fit per-phase slopes `a_phi` by weighted least squares on held-out train sequences, apply to
val, and re-score. Report the range of `a_phi`. If the range across phases is under 0.03, the model is
already phase-calibrated and the block is dead. If the closed-form re-scoring moves val by 0.0005 or more,
you have measured the block's ceiling for the price of a regression.

**Falsifier.** Kill if the closed-form gain does not transfer to the learned version (that would mean the
closed form is fitting the shift, which is a val leak, not a block).

### B6 — Robust loss with δ calibrated to the mixed target (loss geometry)

**Mechanism.** This is the block E2 points at. The mix makes the target a compromise, and on rows where
label and teacher disagree it is wrong by `0.5*|y - t|` no matter what β is. Those rows are not `since`-rows,
so the age weight does not touch them, and because the loss weight is still `|y|`, they sit at the top of
the weighting. MSE then spends capacity chasing a target that is known to be wrong there by construction.
A robust loss is the only block on this list that removes a *row-level corruption* rather than a variance
or a bias, which makes it structurally disjoint from both knobs — the predicted sign is positive exactly in
proportion to how much disagreement variance is unexplained by `since`.

**Exact change.** Replace `|y|`-weighted MSE with `|y|`-weighted Huber, `δ = c * MAD_train(p - m)` measured
at the end of epoch 1 (so `δ` follows the mix rather than the label), `c ∈ {1, 2}`; plus a hard clip of each
row's loss at the 99.5th train percentile. Two lines plus one statistic.

**Free diagnostic.** `R^2` of `|y - t|` on `(since, |y|)` from a small GBM, and the `|y|`-weighted loss mass
in the top 0.5% of `|y - t|` rows. `R^2 > 0.3` means the age weight already handles disagreement: kill, or
expect a negative interaction. Loss mass under 5%: kill.

**Falsifier.** Sweep `c` over two values; if the optimum is the same with and without the mix (measure the
labels-only arm too), there is no interaction and this is just a robustness knob. Also kill if `δ` at epoch
1 is more than 3x the epoch-4 residual MAD — that means the loss is de-robustifying itself over training.

---

## 3. A 12-run screen

Twelve 40-minute runs = 8 hours, one machine, one night. A 2^(6-2) fractional factorial needs 16 runs and
even then aliases the two-factor interactions in pairs; at 12 runs, 6 main effects plus 15 two-factor
interactions is 21 parameters, so a resolution-IV screening design does not exist in budget. This is a
**filter**, not an answer, and the confirmatory design for a surviving block is stated below.

**Protocol, all 12 runs.** Same seed per arm where the design says "same seed"; identical val rows for every
arm; identical epoch count (4) and LR schedule; the pair fixed ON (age weight `max(0.25, 1/(1+since/32))`
and β = 0.5, constant — the "rising with distance to the next label change" switch is a level choice to be
tested afterwards, not inside the screen). Score every arm with a **paired bootstrap over val sequences**
(1000 resamples of sequences, not rows) against the arm it is meant to be paired with. An arm is "alive"
only if its delta interval excludes 0 *and* its point estimate is at least 0.001.

| # | arm | seed | what it buys |
|---|-----|------|--------------|
| 1 | pair only | 1 | paired baseline at seed 1 |
| 2 | pair only | 2 | second seed on the incumbent config → the noise ruler |
| 3-8 | pair + one block, six blocks | 0 | six main effects, fully paired to the existing 0.6905 run |
| 9 | pair + A + B | 0 | completes the 2x2 from the A and B runs among 3-8, plus 0.6905 → `I_AB` at seed 0 |
| 10 | pair + A | 1 | |
| 11 | pair + B | 1 | |
| 12 | pair + A + B | 1 | with run 1, gives `I_AB` at a second, independent seed |

`A` and `B` are chosen after runs 3-8: `A` = largest paired delta; `B` = largest delta among blocks whose
predicted mechanism is *different* from `A`'s, unless a same-axis block beats it by more than 0.0005. This
deliberately spends the interaction budget on a cross-axis pair, because a same-axis pair is the one most
likely to come back negative (substitutive) and least likely to produce something worth stacking.

**What the design identifies.**

- Six main effects, each at one seed, paired to the same-seed incumbent. Precise enough to separate "worth
  0.003" from "worth 0.0005", not precise enough to be a claim about seeds — runs 1-2 exist so the reader
  can see how much a seed costs.
- **One** two-factor interaction, `A x B`, and it does so **twice**: at seed 0 (the A and B runs among 3-8,
  run 9, plus the existing 0.6905) and at seed 1 (runs 1, 10, 11, 12). Because each 2x2 shares a seed, the common seed effect cancels in the
  `+1,-1,-1,+1` contrast, so these are much more precise than the main effects. The two estimates give a
  crude seed-variability read on the interaction itself. This is the entire reason the design is built
  around a same-seed 2x2 rather than "six blocks, three seeds each."
- A two-seed read on the best cell if it happens to be `A+B`.

**What it cannot identify — state this before running it.**

- The other 14 two-factor interactions, and every three-factor interaction.
- **The quantity this document is actually about.** With the pair fixed ON, a block's delta is its
  *incremental* gain on top of the pair, which is not the same as its *interaction* with the pair. A block
  can be additive on top of the pair and still show a large delta. Testing that requires the block × (age
  weight) × (mix) cube: 8 cells, 4 of which (base, age, mix, both) already exist, so **3 extra runs per
  surviving block**. That is the correct follow-up, and it is why the screen keeps only one or two blocks.
  Burn 12 runs proving six blocks are additive-with-everything and you have learned nothing about the
  question that motivated the 2x2.
- Any interaction smaller than about 0.001. The design cannot see it, and neither can the metric's own
  seed noise.

**Stopping rules.** After runs 3-8: if no block clears 0.001 paired, skip runs 10-11 and spend 3 of the
remaining 4 runs on the block × (age) × (mix) cube for the single best block, keeping the last for its
second seed — a block that is additive-with-everything at +0.0015 is worth more than a block with a nice
story at +0.0005. After runs 9-12: if `I_AB` at seed 0 and
seed 1 disagree in sign, the interaction is noise at this sample size, and the honest conclusion is that the
pair is additive and the 2x2's +0.0016 was a seed.

**Order of the diagnostics.** Before any run: the `I` bootstrap and shared-seed check, `Var(t | since)` and
the teacher's train residual by decile (E1), the `|y - t|` regression and its loss-mass share (E2), the
per-epoch `I_e` and the four cells' epoch-3→4 curvature (E3). Those four cost minutes and they rank the six
blocks before a single CPU-hour is spent. B5's closed-form calibration and B3's phase probe are also
zero-run, so B3 and B5 can enter the screen with prior probabilities already adjusted.
