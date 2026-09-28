# Current frontier: what can still move the public score

Current measured public champion: `final_since2_gclp.zip`, **0.6622**.
`final_mixed_ema6_shared.zip` scored **0.6616**. The public leader in the
user-provided snapshot is 0.6677. The old 2026-09-15 `frontier.md` is a
historical artifact; its score, validation, and paused-status claims are stale.

## What transferred

| Gain type | Strongest evidence | Current implication |
| --- | --- | --- |
| Exact metric and inference alignment | Clipping the organizer baseline's outputs moved public 0.5935→0.6503 (E-061/069). | Check every candidate with clipped predictions and the exact scorer mask. |
| Selection-aware learning | Propensity squared weighting was a large offline gain; valid mask fine-tuning moved public 0.6586→0.6600 (E-038/124). | The direct mask auxiliary missed its pair gate; keep the proven propensity and fine-tune recipe. |
| Lower estimation variance | Four-epoch EMA moved public 0.6601→0.6618; the corrected since-weight two-GRU pair moved it to 0.6622 (E-154/223). | Favor improvements that reduce repeated outcome noise without more serving work. |
| Independent prediction errors | A scaled organizer baseline helped early GRUs; the mature pair uses two GRUs (E-084/173). | A new member must improve the *pair*, not just its own score. |

## New EDA that changes idea selection

- Class-3 outcomes almost never change without a visible i0 price-code event.
  A next-eight-event oracle reproduces the inferred reference on about 92–93%
  of sampled episodes, but only 63–71% of score weight. The exact undisclosed
  target rule remains unknown. See [event-horizon figure](artifacts/brainteaser_event_horizon.png).
- Two-or-more-tick outcomes are 11.7% of sampled scored rows and 48.0% of
  out-of-fold pair weighted absolute error. Their episode weight effective
  counts are only 100/122 across the two development folds. See
  [error budget](artifacts/brainteaser_error_budget.png) and
  [concentration](artifacts/brainteaser_episode_concentration.png).
- Rows at least 32 steps after a price event carry 59/61% of sampled class-3
  score weight and 65/68% of weighted error. Event-entry snapshots therefore
  misranked a spread projection that failed on exact scored rows. See
  [age/error plot](artifacts/brainteaser_age_error_budget.png).
- All model-based notebook probes were corrected to use fold-specific
  out-of-fold pair predictions. The first in-sample pass was not a valid
  headroom estimate. See [notebook](../notebooks/brainteaser.py).

## Bounded arm results

| Arm | Honest readout | Decision |
| --- | --- | --- |
| Class-3 tier CE auxiliary | Fold-0 pure-chain replacement pair +0.000040 ±0.000374 exact mask, +0.000058 ±0.000350 at λ=.02 (E-233). | Stop. |
| Causal spread projection | Exact scored class-3 rows −0.00238/+0.00002 across folds at 25% projection (E-234). | Stop. |
| Three-feature causal history correction | Initial 64+64 sample +0.00257/+0.00469; all class-3 sequences +0.00176/−0.00056 (E-235). Drift-only −0.00056/+0.00095 (E-236). | Stop posthoc history tuning. |
| Scorer-mask auxiliary | Fold-0 pure-chain pair +0.000168 ±0.000299 at λ=0; +0.000217 ±0.000289 at λ=.02 (E-237). | Stop before fold 1; below the fixed +0.0005 gate. |
| Current future-movement sign BCE auxiliary | Matched fold-0 single +0.000656, but pure-chain replacement pair +0.000468 ±0.000712 exact mask and +0.000443 ±0.000671 at λ=.02 (E-244). | Stop before fold 1; both miss the fixed +0.0005 pair gate and uncertainty is large. |
| Future 8/24-event quote-path MSE auxiliary | Matched epoch-3 fold-0 single 0.674123 versus 0.676718 parent; pure-chain replacement pair **−0.001435 ±0.001050** exact mask and **−0.001331 ±0.001002** at λ=.02 (E-251). | Stop before fold 1; the high-fidelity teacher did not improve a causal forecast. |

## Direction and event-clock audit, 2026-09-27

- **Allocation check:** class 3 contributes 46–47% of clipped target weight
  and has only 0.411/0.436 within-class WP on pure folds 0/1. Replacing its
  pair predictions with a noncausal true-sign oracle, while retaining the
  pair's predicted magnitudes, raises the *global* fold WP by 0.111/0.108.
  The same oracle in any other class raises it by at most 0.029. Replacing
  class-3 magnitudes with true magnitudes while keeping predicted signs
  **lowers** global WP by 0.012/0.019: large outcomes on the wrong side
  hurt. Class-3 direction and confidence are the central research target;
  naive movement-size scaling is counterproductive. See
  `artifacts/brainteaser_class_oracle_allocation.png` and its JSON table.
- On 831 pure out-of-fold development sequences, the pair scores 0.67735.
  A **noncausal** oracle that supplies only the true signs of its two target
  outputs scores 0.84351; true magnitudes with predicted signs score 0.69882.
  Supplying the true half-difference `(y0-y1)/2` while retaining the predicted
  half-sum scores 0.97564. This is strong headroom evidence for the future-price
  direction/movement component. It says nothing about how predictable that
  component is from available inputs. See `artifacts/brainteaser_oracle_headroom.json`.
- For class 3, the pair already gets 75.1/76.1% of weighted future-movement
  directions correct and weighted direction AUC 0.793/0.804 on folds 0/1.
  Accuracy falls to 71.4/72.0% on rows at least 32 steps after an event;
  correlation between its absolute movement forecast and the realized
  absolute movement is only 0.118/0.122. See
  `artifacts/brainteaser_direction_vs_size.png`.
- The class-3 forecast's medium absolute-contrast band 0.4–0.8 contains
  50–51% of scored weighted contrast error while its weighted direction
  accuracy is 76–77%. The very confident band above 0.8 has 90–91%
  direction accuracy and only 11–12% of error. This favors improving the
  conditional path in medium-confidence, long trend-break regimes over a
  blanket hard classifier threshold. See
  `artifacts/brainteaser_confidence_error.png`.
- **One-event information gap:** on all scored rows in 273 pure class-3
  sequences, the current OOF pair's contrast has weighted correlation
  0.460/0.478 with the current target. If one substitutes the *same trained
  pair's output at the next book event*, correlation becomes 0.755/0.733
  and within-class WP rises 0.415→0.729 / 0.436→0.708. At episode entry,
  matching the current target to the next event's pair output gives
  0.754/0.731 correlation versus 0.165/0.205 at the current event.
  **This is noncausal:** the next event supplies future input. It diagnoses
  where the predictive information arrives, and argues for a direct
  next-book-event side/path auxiliary over a blind attempt to advance the
  GRU output by one row. See
  `artifacts/brainteaser_scored_next_event_oracle.png` and
  `artifacts/brainteaser_event_prediction_lag.png`.
- In the 128-sequence episode snapshot, the **first decoded future mid-price
  jump side** agrees with the target's inferred direction on 98.55/98.56%
  of scored episode weight after excluding zero moves. A 3,000-episode
  check found that the inferred reference-price sign matched the actual
  target-contrast sign in all 2,908 nonzero cases. The decoder was fitted
  on train and the snapshot is sampled, so this is a strong mechanism probe,
  not an exact target-law proof. An opposite-fold HGB using current raw
  features, the pair forecast and 1–64-event history gets next-jump AUC
  0.7590/0.7352 versus the pair's 0.7592/0.7223 on folds 0/1. Its
  target-side AUC is 0.7512/0.7238 versus 0.7522/0.7085 for the pair.
  The gain is confined to fold 1; this screen does not justify another
  next-side auxiliary fit. See `artifacts/brainteaser_next_event_hgb.json`
  and `artifacts/brainteaser_next_event_target_link.json`.
- Across 270 class-3 pure sequences, contrast residual ACF is 0.984 at row
  lag 1 and 0.728 at **book-event** lag 1. It falls below 0.5 after about
  90 rows but only 3 book events. The event-clock ACF falls below 0.1 at 17
  events. Top-quartile absolute-error runs are usually 1–2 events (99th
  percentile 14 events / 193 rows). This is largely overlapping future labels,
  so an AR model of *realized* residuals has no direct serving input. See
  `artifacts/brainteaser_residual_timescale.png`.
- Welch spectra show six-event and three-event peaks in both targets and model
  outputs. A cross-fitted six-phase contrast correction on **all scored rows**
  gains just +0.000057 at 25% shrinkage; full correction loses 0.000287.
  The cycle exists but phase readout is currently a weak score lever. See
  `artifacts/brainteaser_spectral_phase.png` and
  `artifacts/brainteaser_phase6_gate.json`.

The gated fold-0 model finished below its matched parent as a single
(0.675568 versus 0.676718). Its soft gate changes the pure-chain pair by
−0.000499 at λ=0 (−0.000426 at .02); making the same trained classifier a
strict sign gate loses −0.012041 at λ=0 (−0.014537 at .02). Stop this
architecture at fold 0. The strict gate destroys useful uncertainty near a
50/50 direction probability. A standard direct-output GRU with a *training-only*
BCE head for the **current target contrast sign** was the next bounded screen:
weight 0.1, same seed and recipe, same +0.0005 pair gate. This differs from
the existing next-target-change auxiliary because it supervises the current
future-price direction directly without constraining the continuous output.
The direct-sign auxiliary raised the single-model score but its fold-0 pair
lift was +0.000468 ±0.000712 at λ=0 and +0.000443 ±0.000671 at λ=.02. Both
miss the predeclared +0.0005 gate, so it stops before fold 1. The separate
sign logit does rank class-3 directions better than the incumbent pair on
fold-0 pure chains: weighted AUC **0.8243 versus 0.7932** on 345,672
nonzero-contrast scored rows. That ranking is not a score gain. Replacing
25% of the pair contrast with the classifier's soft sign lowers exact-mask
global WP by 0.000591. A chain-cross-fitted correction using only the
classifier's residual signal changes WP by **+0.000002** at the exact mask
and **−0.000007** at λ=.02; fitting a full pair-plus-logit readout loses
0.00499 exact-mask WP because calibration shifts between chain halves.
  The classifier carries some directional information but currently provides
  no reliable additional Pearson signal. Its weighted direction accuracy is
  0.7514 versus the pair's 0.7507 despite the larger AUC difference. A
  fixed abstention gate requiring an opposite-sign classifier logit
  of at least 2.2 and pair contrast below 0.4 has **zero eligible scored
  weight**; there is no high-confidence conflict to exploit at that
  threshold. See
  `artifacts/brainteaser_aux_sign_readout.json` and `scripts/probe_aux_sign.py`.

## The periodic path and the peaks that matter

- Across 20 sampled sequences per class, exact recurrence of the **raw** i0
  price code at lag six book events is 69.6% for class 3, versus 4.6/12.4/12.5%
  for classes 0/1/2. This validates that the six-event spectrum is in the
  observable process, not only a tick-decoder artifact. Its class-3 mid-return
  lag-six ACF is 0.693 on 127 sampled sequences. See
  `artifacts/brainteaser_raw_code_recurrence.png` and
  `artifacts/brainteaser_book_return_acf.png`.
- The six-event oscillator can break into directional paths. In two selected
  trajectories the target jumps before the visible quote trend, while the
  OOF pair stays near its prior. The figure is descriptive and deliberately
  selects high-error scored rows; it is not an estimated prediction opportunity.
  See `artifacts/brainteaser_example_traces.png`.
- A run is defined by event-start contrast residual above that sequence's
  75th percentile of absolute residual. On 273 pure class-3 sequences, only
  2.8/2.9% of these runs last at least six events, but those runs contain
  93.7/91.9% of **all scored weighted contrast absolute error** in folds 0/1.
  Error-weighted median run length is 24/20 book events or 819/678 rows;
  the unweighted median is only one event. This reconciles the short event
  ACF with the long scorer-relevant error episodes. See
  `artifacts/brainteaser_peak_length_scored.png`.
- Event-gap lengths show a seasonality too. A 12-lag linear AR forecast fitted
  on one fold explains 25.7/25.5% of next log-gap variance on the other,
  versus 3.4/3.4% for AR(1). On scored rows the AR(12) forecast has weighted
  partial correlation 0.143/0.152 with the realized next log-gap after
  controlling for elapsed age. This *observable* timing signal changes the
  exact global pair WP by only +0.000117 at 25% correction shrinkage and
  +0.000176 at 50%; below the +0.0005 promotion gate. See
  `artifacts/brainteaser_duration_seasonal_ar.json` and
  `artifacts/brainteaser_timing_pair_global.json`.
- A nonlinear warning statistic made from three recent deviations from the
  six-event return cycle has only −0.033/−0.059 weighted correlation with
  the pair's residual on sampled scored rows; large deviations carry about
  their share of error. That detector does not support a new hard regime gate.
  See `artifacts/brainteaser_cycle_break_signal.json`.
- On all 273 pure-chain class-3 sequences, a causal count of raw price-code
  mismatches from six events earlier has weighted partial correlation only
  0.015/0.078 with absolute contrast error after controlling quote age and
  pair magnitude. Lagging the count by one event gives 0.015/0.092 across
  folds. The mismatch signal misses the preregistered 0.08-in-both-folds
  screen; class-3 error bursts are not consistently announced by a simple
  oscillator break. See `artifacts/brainteaser_raw_cycle_break.png`.

The class-3 bid and ask move together around a bounded, code-dependent spread.
Their levels are a common price component plus this known spread. A VECM would
mostly re-express spread adjustment and lagged price returns already examined
here; the next forecastable quantity is the common future price path, not a
second independent integrated level. Test a VECM only if an out-of-fold
observable return innovation adds residual information beyond the pair.

In a 20-sequence decoded-book diagnostic, common-mid displacement variance
grows from 0.17 at one event to 1.76 at 192, while ask–bid spread displacement
variance remains between roughly 0.23 and 0.69 over those lags. This is
consistent with one drifting common component plus bounded spread, though
the decoder and sample are limited. See `artifacts/brainteaser_vecm_diagnostic.png`.
On retrospective residuals, a fold-crossed ARMA(1,1) removes most linear
lag-one correlation (innovation ACF about 0.004) but leaves squared-error
correlation about 0.159 and the six-event spectral peak. A grid-fit
GARCH(1,1) reduces standardized squared-error lag-one ACF near zero and
improves held-out Gaussian log score versus constant variance; EWMA does as
well on one fold. Those models condition on previously realized *target
errors*, unavailable at test time. A past-quote-volatility proxy has weak,
opposite-sign error-variance correlations (+0.014/−0.097) and does not justify
an uncertainty gate. See `artifacts/brainteaser_arma_garch_residual.json`,
`artifacts/brainteaser_garch11_residual.png`, and
`artifacts/brainteaser_causal_volatility.json`.
ARMA(1,1) one-step residual MSE is 0.0199/0.0249 across folds, below the
ARIMA(0,1,0) random-walk differencing baseline's 0.0227/0.0283. The
differenced residual lag-one ACF is −0.176, so differencing this error process
is a poor default. In contrast, the *decoded price level* has a drifting
common component, for which differencing into returns is natural.

## Next questions

1. Can a train-fitted book-queue or trade signal warn of a **long trend-break
   run** before the target jumps? The simple seasonal return-break and
   past-volatility proxies failed, and the observed future-quote auxiliary
   also lost. Require an out-of-fold partial residual correlation ≥+.08 in
   both pure folds before another GPU fit.
2. Could a branch-conditioned soft payoff mixture capture asymmetric
   stopped-branch outcomes better than the failed symmetric sign×size head?
   Keep the metric-aligned main loss and estimate expected output using
   calibrated side probabilities, without a strict sign gate. It needs a
   matched one-fold pair screen before native inference work.
3. Any candidate that passes two pure-chain folds must then pass exact-mask
   reserved-chain and serving-runtime checks. The public-vs-local calibration
   is unstable; the recent mixed pair looked positive at λ=.02 but tied at
   λ=0 and lost 0.0006 public.

Detailed measurements and source caveats: [brainteaser iteration](2026-09-27-brainteaser-iteration.md),
[idea filter](ideas/2026-09-27-brainteaser-synthesis.md),
[experiment ledger](experiments.jsonl).
