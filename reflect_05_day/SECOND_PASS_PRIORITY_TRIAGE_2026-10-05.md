# Second-pass priority triage - October 5, 2026

Read-only audit of the current October 5 handoff, submission records, ledgers,
and prior review notes. I ran only local scoring/calibration commands over
existing cached predictions; no fit, package build, upload, provider retry, or
new experiment admission was started.

## Main judgment

The previous agent is not stuck because there are no ideas. It is stuck because
most new ideas have been evaluated on weak or repeatedly exposed proxies, while
the real bottleneck is transfer into the complete public-winning predictor. The
current best public score is the seed-4 complete combo at 0.6660. The 0.7000
honest expected-public milestone remains unsupported.

The next window should not run another broad tiny-panel sweep. It should either
settle cheap existing transfer probes with exact archive/runtime evidence, or
run a very small number of complete-predictor comparisons that keep the current
recipe's parent/head procedure.

## Correction to the untracked independent note

I reproduced the untracked `INDEPENDENT_ELEPHANTS_2026-10-05.md` scan:

```text
combo_s4                   holdout 0.70169  publicmix 0.67478  cls [0.8176, 0.7667, 0.7460, 0.4326]
ab85_ema6s4_gclpall        holdout 0.70404  publicmix 0.67744  cls [0.8199, 0.7656, 0.7442, 0.4394]
s4+0.5*ab85_ema6s4_gclpall holdout 0.70422  publicmix 0.67750  cls [0.8201, 0.7674, 0.7466, 0.4383]
```

That makes `ab85_final_ema6_s4_gclpall` a real cached artifact worth auditing.
But it is not a clean holdout win. `scripts/ft_gclp.sh` says `finalall` uses
`--include-heldout-valid` and comments that the holdout score is in-sample. The
September 28 note says full-valid fine-tunes are "unmeasurable locally by
design." So the right interpretation is:

- `ab85_final_ema6_s4_gclpall` is a possible low-cost public-transfer probe, not
  a validated replacement for the incumbent.
- Its local 0.70404 should not be compared to seed-4's excluded-panel 0.70169 as
  if both were out-of-sample.
- Old all-valid packages (`final_e6x2_base25_all`, `final_keep_k4_all`,
  `final_routed_k4_all`, `final_since2_all`) exist locally but are absent from
  the current public registry. They should be inventoried by exact ZIP hash,
  runtime/parity status, and whether they were ever uploaded before spending a
  public submission.

One more calibration caveat: `research/lb/lb_probe.py fit` currently reads a
stale `holdout_stats.json` in this checkout and fit only 12 older submissions
when I ran it, not the current full registry. Rebuild or replace that artifact
before using its LOO RMSE or shift as a decision gate.

## Elephants in the room

1. **Transfer is the missing stage.** E488, E478, E467/E469, target mechanics,
   and activation work are useful, but mostly on H32/H64 or small exposed
   panels. The public-winning object is a complete GRU256 recipe with teacher,
   EMA, Muon/chunk1000/stopped-bootstrap, and final head procedure.

2. **The supplied validation level is not a forecast.** Old D256 beats seed 4
   on the full supplied validation file while losing publicly. Most validation
   rows entered fitting, and the excluded 368 are historically exposed. Any
   "0.70 local means 0.70 public" reasoning is broken.

3. **Some feature and head negatives used degraded controls.** E415's capacity
   control loses about 0.038 WP while several feature families recover about
   0.007 against that damaged control. E421's fitted WMSE control collapses far
   below the unchanged predictor. E468's "regression" anchor is not a preserved
   parent. These tests reject the fitted candidates, but they are weak evidence
   against the mechanisms.

4. **Public-mix and class 3 risk are underweighted in ranking.** The `PUB143`
   class mix used in the independent scan is an inferred scenario, not an
   observed public composition. Still, plausible public-mixture scenarios put
   heavy weight on class 3, the weakest class, where difference-mode residuals
   dominate. Future comparisons should report pooled WP, several public-mix
   proxies, per-class WP, and target slices.

5. **Operational failures are being counted too easily as scientific negatives.**
   W384 is RAM-censored, not refuted. E462 failed ABI/linkage before score, not
   prediction. E419's provider error happened after scientific exports. E463 TPU
   remains queued, not negative.

6. **Some negatives were too broad.** E468 rejects a one-pass frozen coarse
   codebook head whose regression anchor also lost. The hard event-output hold
   rejects output freezing, not event-conditioned belief updates. E426 tested a
   teacher-clock consistency treatment, not target-specific causal input clocks.

7. **The public leaderboard is both the best and most dangerous instrument.**
   It is the only true transfer measurement, but using it to tune +0.0001 blends
   will overfit quickly. Use it for pre-registered, mechanism-distinct probes
   with exact archive hashes, not as an interactive optimizer.

8. **Champion target construction has not been stress-tested enough.** In the
   stopped-bootstrap path, long surviving quiet spans can replace the realized
   label in the regression target with teacher targets. The public recipe is
   strong and should remain the control, but a small original-label reanchor on
   exactly those rows is a direct test of a champion-specific failure mode.

## Ideas rejected too broadly

Keep these alive, but only with tight controls:

- **Fixed GELU/LeakyReLU readout mixture.** E488's fixed 50/50 mixture beats
  GELU at epoch 3 in all three seeds on the small panel. The recurrent context
  gate contributes almost nothing. Transfer the fixed mixture first; do not add
  a dynamic gate yet.
- **Bounded recurrent candidate shape.** E478/E483 show a repeated shape signal,
  but E473/E481 show local slope/cubic controls can explain much of it. Transfer
  one placement with a slope/curvature matched control.
- **Event-aware belief updates.** The current-event clock is nearly necessary
  for scored target changes, and quiet-run updates help class 3. Train a causal
  event/age/innovation residual path; do not freeze predictions.
- **Learned memory as a residual branch.** Learned diagonal decay replicated in
  E467/E469, but the small SSM is weaker than the small GRU. Try compact memory
  alongside the strong GRU, with fixed-decay and equal-capacity controls.
- **Joint target posterior with continuous residual.** Difference-mode residual
  energy motivates a joint model, but the old E468 implementation is a narrow
  negative. Require a competitive pure-regression anchor and train-only support.
- **Anchored feature screens.** Target clocks, event-age, cycle smoothing, and
  related features should be retested only after the fitting procedure can
  preserve the parent and recover a withheld predictable correction.
- **Original-label reanchoring.** Reopen the stopped-bootstrap target path with
  one fixed small label anchor on surviving spans, plus unchanged-bootstrap and
  bootstrap-off controls.
- **Label-aware scoring-selection weights.** E103 used a mask-AUC gate, but the
  metric is weighted Pearson. Reopen as a cheap diagnostic on calibration and
  metric-weighted residual moments before any student training.
- **W384 / capacity.** Still unanswered. Fix loader/RAM behavior before another
  fit; do not call previous provider stops width evidence.
- **Volume and target-clock features.** E413's volume result was one fold/seed
  with a positive point and interval crossing zero. H-084 target clocks were not
  tested by E426. These are second-tier complete-recipe tests, not dead ideas.
- **Cadence and distilled normalizer.** E430/H073 remain only drafted or
  calibrated. Chunk1000 effects are still confounded with optimizer cadence,
  EMA count, and numerator/denominator pooling.
- **QAT/INT8.** Accuracy preservation was near-null, but that is useful. Treat
  it as an ensemble/runtime enabler, not as a score-improvement hypothesis.

## Ideas to keep rejected for now

- Hard output hold between events.
- The exact E468 frozen coarse-codebook head as implemented.
- E464 specialist/adapter implementations as complete-predictor replacements;
  none beat the seed-3 combo on the exposed screen.
- Warmup/cosine schedule as a robust gain; it reversed in E480.
- Same-recipe seed-3/seed-4 bag as a current submission; local excluded-panel
  evidence is below seed 4 and it lacks public transfer.
- Generic post-hoc scaling/calibration/smoothing grids and tiny +0.0001 blend
  searches.

## Recommended priority order

| Priority | Work | Why first | Decisive test |
|---|---|---|---|
| P0 | Evidence eligibility, calibration, and inventory cleanup | No GPU. Prevents bad decisions from stale/stale-ish artifacts and damaged controls. | Rebuild current public registry stats; mark all `finalall/gclpall` local scores as in-sample; rank cached finals by pooled, mixture-scenario, class, and deployment status; flag every screen where the control fails to preserve the parent. |
| P0.5 | Existing all-valid/public-transfer probe audit | Could answer a real "was an old package abandoned?" question cheaply, but only if hash/runtime/parity are already good. | For `ab85_final_ema6_s4_gclpall` or an existing all-valid ZIP: exact archive, one-CPU runtime, ABI/parity. Public upload only as a pre-registered transfer probe, not because local holdout is clean. |
| P1 | Transfer E488 fixed readout mixture to the complete recipe | Strongest repeated small-panel positive with a simple mechanism. | GRU256 current recipe, unchanged control, fixed mixture arm, same final-head procedure, at least two seeds or one seed plus a reserved confirmation. |
| P2 | Original-label reanchoring in stopped bootstrap | Directly targets a champion-specific supervision issue and is cheaper than another architecture sweep. | One fixed reanchor coefficient versus unchanged bootstrap and bootstrap-off, with identical teacher caches and final-head procedure. |
| P3 | Event-conditioned belief update / residual adapter | Best structural clue from target mechanics; distinct from activation tuning. | Head-only strong-state residual first: event/age/innovation versus row-age and shuffled-event controls; then one end-to-end continuation only if it wins. |
| P4 | Finish cadence/normalizer and label-aware selection diagnostics | Explains core recipe mechanics and cheaply checks a rejected metric-weighting idea. | E430/H073 matched scoring; for selection, feature-only versus label-aware selectors on calibration and metric-weighted residual moments before any training-weight change. |
| P5 | Width, learned memory, privileged teacher, volume replication | Plausible but blocked, costlier, or lower-confidence. | One fixed treatment each, with parent-preserving controls and complete final-head endpoint; promote only if it beats controls on a second block/seed. |

The best next scientific bet is still an evolving conditional-belief model:
event-aware state, quiet-run innovations, and compact memory, evaluated against
the current complete predictor. The best next operational bet is to clean up
public-transfer instrumentation and old all-valid package status before burning
more compute.
