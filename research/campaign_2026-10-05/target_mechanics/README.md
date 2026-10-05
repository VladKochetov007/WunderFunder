# Bounded target-mechanics EDA

This is descriptive evidence on historically exposed validation, not a fresh
generalization estimate. Raw evidence and the first fixed-predictor comparison
were saved before reading `oct5_ideas.md` or older mechanism reports. The later
causal event-hold control overlaps a historical idea; that overlap is disclosed,
and the raw cohort and matched cheap comparison remain independently auditable.

## Cohort and provenance

`cohort.json` freezes 40 validation sequences: six fit and four test sequences
per cached label-derived class, each in a different connected component of the
feature-only successor graph. Groups are ordered by a fixed SHA256 string; the
legacy protected hash holdout is excluded. Fit/test have no component overlap.
Class stratification uses historical label-derived metadata **only for selection
and reporting**, never as a feature. All predictors train together across classes.
This reduces dependency within this probe, but does not erase historical exposure
or make four sequences per class representative. Class2 has only 1,365 scored
test rows, versus 9,575/8,232/7,818 in classes0/1/3.

Every raw read is a selected-column read of one 20,000-row parquet group.
Models fit every twentieth row starting at128, without `is_scored` fit filtering.
Evaluation uses the same-origin `need_prediction & is_scored` mask. Input lags
are left-padded within each sequence and never cross sequence boundaries.
Current/past-only inputs are 20 anonymized observed columns, differences at
lags1/8/32/128, and age/current occurrence of changes in `i0_p0` or `i0_p11`.
No physical best-bid, spread, instrument-ordering or horizon interpretation is
assumed for those column names.

Commands use `scripts/limited`, `WNN_SLICE=wnn-campaign.slice`,
`WNN_CPU_QUOTA=200%`, `WNN_THREADS=2`, `WNN_MEM_MAX=2G`,
`WNN_RUNTIME_MAX=600`, and `PYTHONPATH=.`. The main program is
`eda_target.py`; the algebra/figure program is `algebra_probe.py`.
Ruff formatting and checks passed. `resource_sample.txt` records a running
sample of the repeated main run; it is not advertised as its final maximum.

## Empirical mechanics and falsifiers

1. Targets lie on small quantized supports and repeat in runs. Across ten
   sequences per class, individual support sizes are roughly 41–90 in class0,
   24–40 in class1, 9–21 in class2, and7–15 in class3. These are *per-sequence*
   finite samples, not proof of a universal finite target alphabet. Class3 has
   zero fractions36.0% for t0 and32.9% for t1; zeros have zero official weight.
2. t0 and t1 are strongly anticorrelated within these sequences. Mean ordinary
   sequence correlations are -.5545/-.6724/-.8443/-.8803 for classes0/1/2/3;
   after clipping both to ±2 they are -.6733/-.7616/-.8971/-.8980. These are
   structural diagnostics, **not** the official pooled weighted metric.
   The difference mode varies more than the sum mode; mean standard deviations
   `(sum,difference)` are (.939,1.694),(.754,1.635),(.560,1.802),(.391,1.452).
3. Almost every joint target change coincides with a current observed-column
   event, and also with an event one row into the future. The latter diagnostic
   is explicitly nondeployable. Conditional event-at-target-change frequencies
   are1 in classes0/2/3 and approximately.9997 in class1. This establishes a
   nearly necessary clock condition in this cohort, not a sufficient cause or
   a universal law. A causal model still cannot copy yesterday's true label.
4. A simple linear reference decoder is contradicted. Regressing target
   increments on the two observed-column increments separately per sequence
   explains only20/9%,20/22%,35/30%,62/59% of uncentered increment energy for
   t0/t1 across classes. Exact residual hits at1e-5 are essentially absent.
   Current p0/p11 linear fits explain the *sum* mode R² .118/.118/.154/.376,
   but the *difference* mode R² .088/.018/.003/.019. These are in-sample algebra
   audits, deliberately optimistic; they cannot support a deployable decoder.
5. Previous-target persistence yields official pooled WP .99075, but is a
   **nondeployable oracle**, since inference receives no previous target.
   It establishes that most error originates in estimating the latent run
   value and/or run transitions, not simply that a lagged feature should win.

Raw sequence-level support/run/event statistics are in `raw_evidence.json`;
coefficient and residual checks are in `algebra_evidence.json`;
`eda/target_mechanics.png` shows a fixed first sequence per class, without
selecting the most visually persuasive case.

## Cheap predictor control

All scored rows pool into the official `global_wp_clip2_v1`; clipping and weights
come from `utils.weighted_pearson`. Initial fixed scores on26,990 scored rows:

| Predictor | t0 | t1 | mean |
| --- | ---: | ---: | ---: |
| Direct histogram regression | .52802 | .54140 | .53471 |
| Predicted sign × predicted absolute magnitude | .51391 | .49574 | .50482 |
| Ridge current/past inputs | .50180 | .46910 | .48545 |
| Ridge differences/event age only | .40322 | .31223 | .35772 |
| Direct histogram with causal event hold | .49997 | .51119 | .50558 |
| Previous true target oracle, nondeployable | .99146 | .99005 | .99075 |

The direct model uses60 boosting iterations/target; the factor model uses30 for
sign and30 for magnitude at the same12-leaf setting. Both use target-absolute
sample weights with a1e-3 floor; ridge alpha100 is fixed. No hyperparameter or
blend search was performed. Equal tree count is a coarse cost match, not equal
expressivity or a strong-GRU comparison. Naive independent sign/magnitude
factorization loses about.030 pooled WP, although its class2 diagnostic improves.
That class gain cannot be added to other class scores. Class3 scores of these
deployable controls are weak (.09–.162), so they are not a substitute for the
incumbent class3 mechanism. First scores are preserved in
`predictor_scores_blind_v1.json`; `predictor_scores.json` additionally contains
the strictly causal event-hold control, using each sequence's last event row.
Hard event holding loses .02913 pooled WP despite the nearly necessary observed
event clock. Run persistence therefore does not justify freezing a predictor:
its estimate can improve from new information while the unknown true run label
stays fixed. This falsifies the simplest deployment rule on this cheap control;
it does not automatically falsify a trained transition/posterior mechanism.

## Two deployable architectural hypotheses

**Joint quantized conditional distribution plus residual.** Keep a causal
recurrent trunk; learn a latent regime from current/past observations (never the
label-derived class). Predict a joint distribution over a *train-defined*
clipped target grid, plus a small continuous residual. This differs from an
unconstrained two-output regression by imposing a joint distribution and support
prior, not merely by rotating coordinates. Decode each target using
`E[abs(Y) * Y | X] / E[abs(Y) | X]`, with a defined zero-denominator behavior,
because absolute-target weighted MSE targets that conditional quantity. An
ordinary probability mean need not optimize the weighted objective. Strong
anticorrelation and compressed sum mode motivate coupled states; they do not
justify a physical spread decoder. Compare against a same-trunk/capacity direct
WMSE head and an unconstrained sum/difference head, same optimizer/head procedure.
Falsify if cross-component gains fail, if gains come solely from class routing,
or if valid unseen amplitudes systematically hurt the constrained head. Cost:
one extra modest output layer; no second trunk or inference future context.

**Event-conditioned latent run filter with target-specific updates.** Maintain
a posterior state representing the unknown persistent pair. Causal observed
events trigger a transition branch; row-by-row volume/trade information updates
the posterior within a run. Output a target-specific correction/residual, with
an auxiliary stationarity penalty only on observed no-event transitions, tested
against shuffled event masks and an equal-capacity ungated branch. A hard
inference hold is the zero-training-cost falsifier; it may lose because fresh
within-run evidence can improve a posterior even when the true label is fixed.
Thus output freezing and hidden-state freezing are distinct interventions.
Do not infer t0-row/t1-event clocks from anticorrelation alone: the measured
t0/t1 change rates are often similar, and both need paired target-specific
controls. Falsify if a causal event branch fails on strong complete-predictor
residuals or a sham mask matches it. Cost: small transition/update heads sharing
the trunk, matched parameter count and complete fixed-head training procedure.

These are hypotheses with deployable information boundaries, not claims of a
score improvement. The missing evidence is whether a strong complete predictor
already models joint support, how residual errors split between mode/value and
transition timing, and whether any structured head survives matched complete
predictor tests on additional components/seeds.
