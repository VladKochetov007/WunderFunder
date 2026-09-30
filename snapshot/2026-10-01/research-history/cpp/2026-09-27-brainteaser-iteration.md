# Brainteaser iteration: event path, rare payoff tier, and deployment parity

## Measured public anchor and why this iteration exists

The user-reported public champion is `final_since2_gclp.zip` at **0.6622**. The newer
`final_mixed_ema6_shared.zip` scored **0.6616**. The old pair's local reserved score
was 0.696577 at the exact mask; the mixed pair's was 0.696560 (delta
−0.000018 ± 0.000776 by chain bootstrap). The previously attractive +0.000653
comparison used a softened mask, λ=0.02. It was not an exact-mask gain.

Historically, four types of changes moved the public score: exact scorer and
deployment parity (+0.0568 from fixing clipping), training on states favored by
the scoring selection, reducing fit variance with EMA/fine-tuning, and mixing
strong members with different errors. The mixed result says a small softened-mask
gain is insufficient evidence for a submission. See
[`research/2026-09-27-final.md`](2026-09-27-final.md) for the measured history.

## New event-clock EDA

All analysis uses visible validation labels. The price-code transition decoder
in `class3_moves_full.pkl` was fitted on train sequences. The event probe uses
128 randomly selected class-3 *development* sequences, 64 from each of two
pure-chain folds, one first-scored snapshot per i0 quote episode (54,722
episodes). The row plot uses all 299,523 scored rows of those sequences. No test
data or leaderboard feedback enters the probe.

- In a separate 24-development-sequence sample per class, virtually all target
  changes coincided with i0 price-code changes. Class 3 had **0** quiet-book
  target changes among roughly **109,000** target changes. So a hidden row-clock
  target reset is not the missing mechanism.
- The two target values independently imply the same latent class-3 future
  reference price on **99.995%** of eligible episodes. This validates target
  inversion as a diagnostic. The inversion and future quote path are **not
  available as prediction inputs**.
- Averaging the next eight visible quote mids reconstructs that reference more
  often than averaging four: exact episode rates **93.1% / 92.1%** at eight
  events on folds 0/1, versus **89.9% / 88.5%** at four. Exact *metric-weight*
  rates are **70.5% / 63.5%** versus **54.5% / 45.3%**. Longer averages near
  20–24 events reduce weighted absolute tick error further while losing exact
  episode matches. **Eight is not the discovered label formula**; the tail
  remains poorly reconstructed.
- A winning tier of at least one tick occurs on only **10–12% of scored book
  episodes** but carries **63–66% of their clipped target weight**. Higher-tier
  episodes have more scored rows as well as larger `|y|`: median scored rows
  per episode rise from 1 at touch/inside to 2–3 for positive tiers.
- On scored rows, weighted probability of a two-tick winner rises with causal
  time since the last price event. Yet the pair's direction magnitude falls on
  the longest quiet stretches, and its side accuracy also declines there
  (especially fold 1). A high tier alone does not justify scaling the signed
  prediction. The future rows-to-next-event curve is an oracle diagnostic.
- At the same causal age, scored rows have a higher two-tick prevalence than
  unscored rows, especially in fold 1. After additionally matching bins of the
  existing `prop_v1` scorer-propensity model, the gap is **−0.05 percentage
  points** on fold 0 and **+1.48 points** on fold 1. Much of the raw association
  is already captured by propensity, and the residual is unstable. This
  weakens a tier-specific rationale for another mask head without ruling out
  broader representation benefits. See the
  [`mask-tier plot`](artifacts/brainteaser_mask_tier.png) and
  [`age/propensity-matched table`](artifacts/brainteaser_propensity_gap.json).
- In the same 128-sequence scored-row sample, realized tiers ≥2 carry **48.0%**
  of the pair's clipped-target-weighted absolute error while representing
  **11.7%** of scored rows. Correct-side ≥2 rows alone carry **28.9%** of that
  error; wrong-side ≥2 rows carry **19.0%**. One-tick outcomes carry another
  **36.7%**. This is an error-budget diagnostic, not an attainable Pearson gain.

The **out-of-fold** pair tier probe used a small HGB with matched settings on separate
pure-chain folds, trained at one scored snapshot per episode with episode weight
capped at 100. For tier ≥1, adding all current raw features to pair+age raised
weighted AUC from **0.626→0.640** on fold 0 and **0.617→0.631** on fold 1, but
weighted Brier worsened. For tier ≥2, raw features changed AUC
**0.580→0.552 / 0.516→0.522**, with worse Brier on both folds. The full causal
past-event path changed tier ≥2 AUC by **−0.009 / +0.035** and improved Brier
only in fold 1. The small focused past-event-rate/drift model gave tier ≥2 AUC
gains of **+0.019 / +0.042**, with Brier improvements **+0.0001 / +0.0023**.
This is a possible information lead, not a scored-model gain: features were
chosen after inspecting the same two folds, and the tier auxiliary failed its
paired metric gate. The focused readout needs untouched-chain verification;
[`brainteaser_focused_screen.json`](artifacts/brainteaser_focused_screen.json)
records the numbers.

The live [marimo notebook](../notebooks/brainteaser.py) contains the data
construction, probe, and interactive tier-age plot. Evidence snapshots:
[`brainteaser_metrics.json`](artifacts/brainteaser_metrics.json),
[`brainteaser_event_snapshot.npz`](artifacts/brainteaser_event_snapshot.npz),
[`brainteaser_scored_rows.parquet`](artifacts/brainteaser_scored_rows.parquet),
[`event-horizon figure`](artifacts/brainteaser_event_horizon.png), and
[`tier-age figure`](artifacts/brainteaser_tier_age.png). The
[`error-budget figure`](artifacts/brainteaser_error_budget.png) and
[`error-budget table`](artifacts/brainteaser_error_budget.json) show the
side × tier split.

## Deployment path audit

The mixed ZIP serves **two** GRUs on a v3 CPU but only its **first** GRU on its
v2 fallback. The old submitted ZIP serves two on either path. In one matched
400,000-row scorer-image run at one CPU, the old full pair took **63.0 µs/row**
on v3 and **83.3** on v2; mixed took **48.8** on v3/full and **38.1** on v2/single.
The corresponding 20-sequence subset WPs were **0.719760**, **0.720067**, and
**0.717577** (old, mixed full, mixed single). On the 368 reserved sequences
with the exact visible mask, mixed full was essentially tied with old
(−0.000018), while mixed single was **−0.002153**. The public elapsed-time ratio
is 25m52s / 36m27s = **0.710**, between same-host v3/full ratio **0.775** and
old-v3 to mixed-v2/single ratio **0.605**. Runtime and score do **not** identify
the organizer's CPU flags or which path ran. The fallback is material enough to
audit in any future package; it is not an established explanation of this
submission's public loss. Raw measurements are in
[`fallback_runtime_audit.json`](artifacts/fallback_runtime_audit.json) and
[`fallback_score_delta.json`](artifacts/fallback_score_delta.json).

## Idea synthesis from four literature scouts

The scouts read selected material in `~/Documents/Trading/` and compared their
proposals with the experiment ledger. Their book and paper analogies motivate
tests; they do not establish gains on this data.

| Idea | Relationship to a measured gain | Current evidence and decision |
| --- | --- | --- |
| Class-3 tier auxiliary, discarded at inference | Adds training signal without serving cost, like the successful next-change auxiliary; aligns with scorer weight | Run one conservative matched fold. The raw/history snapshot probes caution that the ≥2 tail may be weakly forecastable. |
| Side × continuation or successor-event head | Separates first-passage side from later price path; new event supervision rather than the failed 2nd/4th *target-change* heads | Candidate after tier arm, but target should use 8–24 observable future price events and be checked against actual target inversion. Four events alone miss much high-weight mass. |
| Competing-risk survival head | Could teach event arrival and direction jointly, possibly improving selected-state representations | Earlier age/propensity readouts and unused sign-logit readout failed. Require a causal residual screen before training. |
| Decoded depletion/refill or phantom-wall features | Potential independent information source, analogous to the early successful baseline blend | Raw i0/i1 volume shocks failed both folds. Require train-only decoder stability and residual event-level AUC gate. |
| Better validation under selection/source shift | Replicates the exact scorer and protects against the mixed pair's false local lift | Make λ=0 and pure chains primary. Reweight by causal state and source only as stress scenarios, never as hidden-test forecasts. |
| Joint scoring-mask auxiliary | Directs capacity to selected states, extending the strongest public training mechanism | Plausible second bounded arm after the tier result; the prior propensity model and true-mask fine-tuning already capture much of the effect. |

Relevant local literature includes Cartea–Jaimungal–Penalva, *Algorithmic and
High-Frequency Trading*, §§4.4, 10.4, 12.2 (queue removal/refill and impact);
Harris, *Trading and Exchanges*, ch. 6 and §13.7.2 (queue priority and adverse
selection); Paleologo, *Elements of Quantitative Investing*, §§8.1–8.2
(validation discipline); Nielsen, *Practical Time Series Analysis*, ch. 11
(forecast evaluation); and N-BEATS (past/future decomposition analogy). The
scouts' broader raw proposals are retained in the task thread; any pursued
variant gets a named artifact and a chain-disjoint falsifier.

## Scorer-weight concentration and validation precision

### Correction to model-based EDA

The first notebook pass used predictions from the final submitted pair, whose
fit included the validation rows. That was appropriate for describing the
submitted pair but not for an out-of-fold headroom screen. The notebook was
rerun with `ab62_base_s4_f{k}` and `ab62_since_f{k}` on their held-out folds;
the model-based numbers above and saved artifacts now use those predictions.
Target anatomy, which does not use model predictions, was unaffected.

The live [marimo EDA](../notebooks/brainteaser.py) now includes an episode
concentration plot, saved as
[`brainteaser_episode_concentration.png`](artifacts/brainteaser_episode_concentration.png)
with its [numbers](artifacts/brainteaser_episode_concentration.json). Across
25,386 / 29,336 sampled scored quote episodes in development folds 0 / 1, the
highest-weight 1% account for **42.1% / 44.0%** of target weight. Kish weight
effective counts are only **694 / 731**. Among episodes whose winning tier is
at least two ticks, effective counts fall to **100 / 122** (908 / 1,229 raw
episodes). Episodes from the same sequence can still be correlated, so these
are concentration diagnostics, not independent-sample confidence intervals.
The same 64 sequences per fold have target-weight effective counts of
**36.4 / 40.5**; the heaviest 10% hold **32.7% / 29.9%** of weight
([plot](artifacts/brainteaser_sequence_concentration.png),
[numbers](artifacts/brainteaser_sequence_concentration.json)). These are
sample-specific descriptive counts, not score uncertainty estimates. This
raises the threshold for believing small local gains, especially tail gains:
compare on disjoint chains with the exact scorer mask and inspect contributions
by episode and sequence before promotion.

## Predeclared training gate

The first arm, `--aux-tier 0.1`, adds a four-logit class-3 winning-tier CE
loss, weighted by the existing training selection weight and clipped `|y|`.
The main two-target wMSE, seed 3, EMA, propensity, since weighting, four epochs,
fold 0, and second pair member stay matched to `ab62_since_f0` plus
`ab62_base_s4_f0`. The auxiliary logits are discarded at inference. Promotion
requires at least **+0.0005** paired global WP at **both λ=0 and λ=.02** on
fold-0 pure chains. Otherwise stop. A pass requires fold 1 with the same
hyperparameters, positive lift there and at least +0.0005 pooled exact-mask
lift, then source/mask stress and deployment parity before any final package.

This experiment answers whether a tier *training loss* helps the strong pair.
The event EDA and snapshot classifier do not themselves imply an attainable
submission gain.

**Result (E-233).** At epoch 3 the tier model alone scored 0.676502 versus
0.676718 for its matched parent. Replacing that parent in the fold-0 pair gave
**+0.000040 ± 0.000374** on 432 pure-chain sequences with the exact mask;
at λ=.02 the result was **+0.000058 ± 0.000350** (chain bootstrap SD).
Both miss the predeclared +0.0005 gate, so no fold-1 fit or final package is
justified. The [pair readout](../artifacts/logs/ab75_tier01_f0_pair.txt) and
[training log](../artifacts/logs/ab75_tier01_f0.log) retain the snapshot.

## Event-snapshot trap: spread projection

For class 3, the two outputs share an inferred future price, suggesting a
simple low-cost correction to their sum using the causally decoded spread.
The train-fitted code table covered **99.998%** of sampled quote events. At the
first scored snapshot of each event, a 25% projection seemed to improve
weighted correlation by about **+0.0021 / +0.0046** on folds 0 / 1. Scoring
**every scored row** of the same 128 pure-chain sequences reversed the result:
the exact class-3 correlation changed **−0.00238 / +0.00002** at 25%, and
**−0.00704 / −0.00189** at 50%. We stop the idea here. The stopped branches
do not obey the naive sum constraint, and first-snapshot weighting missed
what the metric emphasizes later in an episode. See the
[`spread-projection screen`](artifacts/brainteaser_spread_projection_sample.json)
and E-234. This also limits what a one-snapshot tier classifier can establish
about a full-row submission.

The [scored-row age/error plot](artifacts/brainteaser_age_error_budget.png)
and [table](artifacts/brainteaser_age_error_budget.json) explain the mismatch.
On the same 128 pure-chain sequences, event-age-zero rows are about **11%** of
scored rows but only **4–5%** of weighted absolute error. Rows at least 32
steps into the current price event contain **59% / 61%** of target weight and
**65% / 68%** of weighted error on folds 0 / 1. The first scored snapshot is
usually age zero or one; it cannot stand in for the late-episode rows that
dominate the score. This anatomy does not imply that training should give late
rows more weight: the existing since weighting already improved held-out folds
and public score by reducing repeated-label fit variance (E-182/223).

## Causal history residual screen

The focused tier probe suggested three price-event history variables: past
eight-event mid drift and elapsed row counts over eight and sixteen events.
A fixed ridge correction to the out-of-fold pair, trained on the opposite
pure-chain fold and evaluated with every exact-mask class-3 scored row, seemed
promising on the original 64+64 sampled sequences (**+0.00257 / +0.00469**
class-3 correlation). Expanding the **same** procedure to all 112 / 98 eligible
class-3 development sequences changed the result to **+0.00176 / −0.00056**.
The [sample](artifacts/brainteaser_history_residual_sample.json) and
[full-fold readout](artifacts/brainteaser_history_residual_3feat_full.json)
are saved. The candidate misses a positive-both-folds gate and exposes another
sample-selection trap. The final simplification to only eight-event drift also
changed sign across full folds (**−0.00056 / +0.00095**); see the
[drift-only readout](artifacts/brainteaser_history_residual_drift_full.json).
Stop this posthoc history branch without fitting more coefficients to these
development folds.

## Next bounded arm: scorer-mask auxiliary

The public gains from propensity weighting and true-mask fine-tuning were
training-selection gains. A candidate training-only binary head can teach the
shared GRU to represent the activity states selected by the scorer. Its label
exists only on validation-source training sequences; train-source sequences
must receive zero auxiliary loss. The head is discarded at inference. Keep the
matched `ab62_since_f0` recipe, seed, EMA, main wMSE, and pair anchor; add a
0.1-weight positive-balanced mask BCE. Use the fixed epoch-3 snapshot to match
the parent, irrespective of which epoch gives the candidate's best local score.
The **same +0.0005 pair lift at both
exact mask and λ=.02** on pure fold 0 is required before fold 1. If it fails,
stop rather than tune BCE weights on the same development fold. This tests
representation transfer from mask prediction, not a direct hidden-mask gate.

## Comparable public solutions and literature

The [Jane Street Real-Time 8th-place solution](https://github.com/evgeniavolkova/kagglejanestreet/blob/master/solution.md)
used time-series GRUs, structurally different ensemble members, extra rolling
targets, and a chronological gap check. Its author reports about +0.001 from
auxiliary targets. The transferable question here is whether an **observable
future price-event average** supplies a cleaner training signal than our
failed second/fourth *target-change* heads; Jane Street's online update gain
does not transfer because this scorer does not reveal test targets online.
The [Optiver Trading at the Close winner's writeup](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution)
combined CatBoost, GRU and Transformer with a held-out late period. This
supports testing genuinely different error sources, but our class-3 GBM
residual overfit and a third model has a real CPU-serving cost. The
[Optiver 14th-place leakage postmortem](https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/clash-royale-14th-place-solution-for-the-optiver-t)
is a useful parallel to the notebook's in-sample-prediction correction:
grouped time differences and out-of-fold predictions are requirements before
interpreting model headroom. [Dixon's event-flip RNN paper](https://arxiv.org/abs/1707.05642)
motivates event-clock supervision, while the existing next-change sign head
already had no residual readout gain (E-232). [Lim and Gorse's joint
direction/size model](https://arxiv.org/abs/2004.01498) motivates a
distributional head, but our plain outcome-code CE and tier auxiliary failed
their metric gates. These are idea sources, not score forecasts for this
competition.

## Event-clock follow-up: direction, peaks, and causal timing

The live [marimo notebook](../notebooks/brainteaser.py) now includes out-of-fold
direction/size oracles, raw-code event recurrence, row- versus event-clock
residual ACF, Welch and Haar spectra, ARMA/GARCH error diagnostics, decoded
quote-return and event-duration AR fits, and actual high-error traces. Its
model-based plots and readouts use fold-specific OOF predictions on pure
chains; label-based oracles are marked noncausal. The official metric is the
[live pooled WP definition](https://wundernn.io/connectome/docs/data_overview),
which supersedes this checkout's older block-average `METRIC.md`.

The class-3 OOF pair predicts the weighted movement sign at 75.1/76.1%
accuracy and 0.793/0.804 AUC, but its absolute movement correlation is only
0.118/0.122. A true-sign oracle raises pooled WP from 0.67735 to 0.84351,
whereas a true-magnitude oracle with the pair's sign reaches only 0.69882.
The real challenge is a calibrated *probability of the future price path* and
the rare large-directional outcomes, rather than a hard sign decision. The
factorized GRU's strict sign gate lost 0.0120 exact pooled WP on fold 0; its
soft gate also missed the pair gate. See E-238 and E-241.

Residual ARMA(1,1) one-step R² is 0.57–0.58 on opposite folds in event time;
GARCH(1,1) removes residual squared-innovation lag-one correlation. This
would be useful only if past realized label errors were available at
inference. The label involves future quote events, and the residual ACF has
fallen to about 0.09 by event lag 20. Normal event-level high-residual runs
last one or two events, but the *scorer-weighted* error concentrates in rare
20–24-event, 678–819-row runs; 2.8–2.9% of high-error runs at least six events
long contain 92–94% of all class-3 weighted contrast absolute error in these
two pure folds. These are the trend-break episodes shown in
[`brainteaser_example_traces.png`](artifacts/brainteaser_example_traces.png).

Raw class-3 price codes recur exactly after six book events on 69.6% of a
20-sequence sample, and decoded returns have lag-six ACF 0.693 on 127 sampled
sequences. A 12-lag seasonal return correction changes sign across folds.
Durations between price events have another periodic pattern: a 12-lag AR
predicts the next log duration with 0.255–0.257 held-out R², but an exact
global-WP correction is only +0.0001–0.0002. [Cont and de Larrard's queue
model](https://arxiv.org/abs/1104.4596) treats next-move side, price-change
duration, and return autocorrelation jointly, while [Engle and Russell's
conditional-duration model](https://eml.berkeley.edu/symposia/nsf98/engle_russell.pdf)
is a useful analogue for irregular event timing. In this dataset, those
analogues have not yet added enough residual information to justify serving
code. A competing-risk side/duration head would need a new observable signal
beyond the current next-change auxiliary and seasonal AR readout.

## Incremental sign information and class allocation

The direct target-sign auxiliary's unused output has weighted direction AUC
0.8243 on 345,672 scored, nonzero class-3 contrast rows in pure fold 0,
versus 0.7932 for the incumbent pair. This ranks direction better but is not
itself a global Pearson improvement. A fixed 25% blend of
`0.5*tanh(logit/2)` with the pair contrast loses 0.000591 global WP at the
exact mask. A chain-half-cross-fitted, 25%-shrunk correction based on only the
logit's component orthogonal to the pair contrast moves WP by +0.000002
(exact mask) and −0.000007 (λ=.02). Full cross-fitted linear calibration
loses 0.00499 exact-mask WP, with unstable intercepts across halves. This
supports retaining the existing continuous outputs and treating classifier
AUC as a diagnostic, not a promotion metric. The reproducible probe is
[`scripts/probe_aux_sign.py`](../scripts/probe_aux_sign.py), with
[`brainteaser_aux_sign_readout.json`](artifacts/brainteaser_aux_sign_readout.json).
At threshold zero the sign head's weighted accuracy is 0.7514 versus the
pair's 0.7507, far smaller than the AUC difference suggests. A fixed
high-confidence disagreement abstention rule (|logit|≥2.2, pair contrast
<0.4) finds no eligible scored weight, so it provides no implementable gate
at that confidence level.

A class-specific oracle allocation on all 831 pure development sequences
finds class 3 carries 47.1%/46.3% of clipped target weight in folds 0/1 yet
scores only 0.411/0.436 within class. Giving true signs to only class 3
raises the global fold metric by 0.111/0.108, far above any other class's
true-sign headroom. Giving class 3 the *true magnitudes* while retaining the
pair's wrong signs instead **lowers** global WP by 0.012/0.019. This is
noncausal diagnostic headroom, not a forecast. It means any magnitude head
must retain soft directional uncertainty; a larger move on the wrong side
can be worse. See the [class allocation figure](artifacts/brainteaser_class_oracle_allocation.png)
and its [numbers](artifacts/brainteaser_class_oracle_allocation.json).

The raw-code lag-six mismatch count gives a more literal causal cycle-break
proxy than return innovation. On all 273 pure class-3 sequences, its weighted
partial correlation with absolute pair contrast error is only 0.015/0.078
across folds after controlling for quote age and predicted move size; using
the previous event's count gives 0.015/0.092. The second fold is suggestive,
the first is not. It fails the predeclared +0.08 correlation screen in both
folds. The [figure](artifacts/brainteaser_raw_cycle_break.png) and
[table](artifacts/brainteaser_raw_cycle_break.json) preserve the bins.

**Application of the autocorrelation result.** The observed six-event quote
cycle and ARMA error law should set a *training hypothesis*, not a serving
ARMA adjustment: past true errors are unavailable at scoring time. The next
credible test would retain weighted MSE, add a small training-only auxiliary
for signed future book displacement over roughly 8 and 24 *book events*,
and report a matched fold-0 pair replacement. Its labels can be computed
from train quotes; inference discards the head. An alternative is a small
event-transition residual-innovation loss alongside the level loss, which
could reduce redundant row supervision. The long error runs warn against
replacing the level loss with pure differencing. Both ideas remain untested;
neither deserves a new submission without positive exact-mask gains on two
pure-chain folds and a serving-time audit.

## The next-event information gap

An event-shift analysis weights each class-3 target at a book-event entry by
the exact scored target mass in its episode. Across 273 pure development
sequences and about 61,500 scored episodes per fold, the pair's contrast at
the *current* entry correlates with that target at only 0.165/0.205. The
same pair's output at the **next** book event correlates at 0.754/0.731.
On all scored rows with a next event, substituting the next-event output
raises class-3 contrast correlation from 0.460/0.478 to 0.755/0.733 and
class-3 WP from 0.415/0.436 to 0.729/0.708. The gap persists through long
quiet stretches: for age ≥256 rows the contrast correlations are
0.323/0.437 at the current row versus 0.802/0.818 at the next event.
Figures: [event-entry shift](artifacts/brainteaser_event_prediction_lag.png)
and [scored-row age](artifacts/brainteaser_scored_next_event_oracle.png), with
[full table](artifacts/brainteaser_scored_next_event_oracle.json).

The future-event output is a **noncausal oracle** using later book input;
this is not evidence that shifting the live GRU state is legal or helpful.
The sharp +1-event peak is consistent with much of the target becoming
visible at the next book update. It weakens the case for an ARMA residual
correction or a pure transition-difference loss, because those would try to
predict an innovation that might have little advance warning. The practical
question is whether a training-only head for *the next observed quote move*
improves the existing GRU's estimate of its conditional distribution before
the event. The existing `aux_next` head predicts the next **target-label**
change, which is a related but different supervised task. Any new quote-path
head should be tested on a matched fold-0 pair before training another fold.

The next-event mechanism can be checked directly in the 128-sequence
episode snapshot. After removing zero first jumps and zero target directions,
the first future decoded mid-price jump has the same sign as the inferred
target reference on **98.55% / 98.56% of scorer episode weight** in folds 0/1.
A separate 3,000-episode sample verified that this reference-side sign equals
the actual target-contrast sign in all 2,908 nonzero cases. The future jump
is unavailable live; this explains the +1-event oracle spike without
claiming a usable lag correction. The decoder is train-fitted and the
snapshot covers 64 pure sequences per fold, so exact target-law claims remain
out of reach.

A 80-tree, 15-leaf histogram gradient booster trained on opposite-fold
causal snapshots (pair outputs, raw features, age and 1–64-event history)
was screened against the incumbent pair. Weighted AUC for the next jump is
0.7590 versus pair 0.7592 on fold 0, and 0.7352 versus 0.7223 on fold 1.
For the current target side, it is 0.7512 versus 0.7522 on fold 0 and
0.7238 versus 0.7085 on fold 1. The positive effect is limited to one fold;
the screen fails a two-fold improvement requirement. Because next-jump
direction and target direction nearly coincide, a standalone side auxiliary
is especially likely to repeat the failed direct target-sign supervision.
The remaining plausible auxiliary is a longer *future path* statistic that
could change the representation without imposing a hard decision at
inference. See [opposite-fold HGB screen](artifacts/brainteaser_next_event_hgb.json),
[target-side link](artifacts/brainteaser_next_event_target_link.json), and
[saved OOF probabilities](artifacts/brainteaser_next_event_hgb_oof.npz).

## Preregistered 8/24-book-event quote-path auxiliary arm

The train-only code decoder reconstructed future event mid paths with just
4,834 unknown transitions out of 17,689,610 train transitions and 82 out of
4,799,217 validation transitions in the cache run. It produces two
training-only labels: clipped target-unit displacement of the average of the
next 8 and 24 decoded book-event mids from the current mid. On scored pure
class-3 validation rows, these noncausal labels correlate with the current
target contrast at 0.951/0.961 for 8 events and 0.930/0.938 for 24 events
across folds 0/1. The 24-event label has lower weighted absolute difference
from the target (0.347/0.329 versus 0.466/0.455), despite lower correlation.
The high fidelity also raises a redundancy concern: it may regularize the
existing target head without adding forecastable information. Cache code:
[`scripts/cache_quote_path.py`](../scripts/cache_quote_path.py); label check:
[`brainteaser_quote_path_label_quality.json`](artifacts/brainteaser_quote_path_label_quality.json).

**Fixed test before outcome inspection.** `ab79_quotepath01_f0` keeps the
`ab62_since_f0` seed, GRU, main weighted MSE, propensity, since weighting,
EMA, next-change and structure auxiliaries. It adds two discarded output
channels and a 0.1-weight class-3 quote-path MSE. The labels are loaded only
for train and the non-held-out validation source; fold-0 prediction uses
neither labels nor future quotes. Evaluate epoch 3 as in the matched parent.
Require at least **+0.0005** pure-chain pair replacement WP against
`ab62_base_s4_f0 + ab62_since_f0` at both exact mask and λ=.02, plus a
chain-bootstrap uncertainty check. If it passes, run the same recipe on
fold 1; otherwise stop. No public package or native inference edit is
motivated by a training-only failed arm.

**Outcome (E-251).** The fixed epoch-3 auxiliary model scored 0.674123 on
the matched full fold, versus 0.676718 for its parent. It improved the
within-class class-3 diagnostic slightly (0.4471 versus 0.4461) but lowered
class 0 (0.8030 versus 0.8060) and the exact global metric. Replacing the
second member in the pure-chain pair lowers WP by **0.001435 ±0.001050**
chain-bootstrap SD at the true mask and **0.001331 ±0.001002** at λ=.02.
It misses the predeclared +0.0005 gate decisively, so no fold-1 training or
native port follows. The high target correlation of future-quote teachers
did not establish *causal* incremental forecast information. Checkpoint and
log remain in `artifacts/runs/ab79_quotepath01_f0/` and
`artifacts/logs/ab79_quotepath01_f0*.log`.
