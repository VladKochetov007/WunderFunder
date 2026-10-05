# Fixed strong predictor: residual mechanics and excluded-row replication

This is a **structural diagnostic**, not a promotion or public-score forecast.
The original 16 component EDA test panel is **entirely fitted by both baseline
models**: 16/16 groups appear in each fitting provenance. Its .7744 WP must not
be presented as an unfitted estimate. The additional 16 groups are drawn from
the exact 368 validation groups excluded from direct baseline fitting. They have
zero direct supervised row-group overlap, but 16/16 parent and 15/16 head graph
components link to other fitted rows, including train data. Both panels are
historically exposed. Neither is an independent fresh holdout; the excluded-row
replication alone cannot prove the mechanism is free of interpolation effects.

The new public milestone is honestly expected 0.7000. Nothing here calibrates
that expectation or demonstrates that milestone.

## Frozen procedure and identities

`strong_residual_probe.py` reads two FP32 prediction files using mmap, selects
only bounded groups, and asserts exact shape 1873×20000×2 and ordered sidecar
row groups 0…1872. Parent is fixed epoch 3, head fixed epoch 0; complete prediction
is float32 `0.5parent +0.5head`, with clipping only inside evaluation/residual
diagnostics. Saved sidecars provide prediction/checkpoint hashes. Official WP
always pools the selected domain, clips both targets and predictions to±2,
and uses each target's absolute clipped value as its weight.

The fitted panel reuses the original fixed EDA test cohort. For the excluded
panel, `strong_baseline_excluded/cohort_fixed_before_array_reads.json` records
selection before prediction-array or raw-parquet reads: first 16 unique graph
components in SHA256 `strong-residual-excluded-v1:component:group` order among
368 excluded groups. No class stratification, winner selection or parameter
change was made. It contains classes0/1/3 and **no class2**; absence was retained.
One group per component prevents within-panel chain duplication, but does not
remove the baseline fitting links above.

Both runs use the 2 CPU/2 GiB parent slice guard. Measured script RSS was below
346,000KiB; measured wall time including plotting was 1.11s fitted/1.34s excluded.
Ruff formatting/checks passed. Raw blind EDA artifacts were not modified.

## Paired descriptive outcomes

| Fixed predictor | Fitted16,26,990 rows | Excluded16,30,419 rows |
| --- | ---: | ---: |
| Parent | .771045 | .740771 |
| Fine-tuned head | .776015 | .743287 |
| Equal complete predictor | .774416 | .742757 |
| Complete with causal event hold | .692929 | .666532 |

Hard holding uses the predictor at the most recent current/past change in
`i0_p0` or `i0_p11`; it can process and retain warm-up predictions and never
looks ahead. It freezes **outputs**, not the original model's hidden-state
updates. The loss persists on the excluded-row panel: ΔWP−.076225, compared
with−.081486 fitted. Class3 complete/hold is .418856/.063571 fitted and
.553540/.274611 excluded. Class2 is .852165/.475262 on the fitted panel only.
This is strong evidence against deploying the proposed hard output hold.
It does not reject a separately trained transition/posterior-update mechanism.

## Scored-transition correction requested by the independent critic

The initial raw event law used all rows. The added audit defines the transition
from row t−1 to t by the **destination** mask `need_prediction[t] & is_scored[t]`.
Its current event is causal; the event at t+1 is separately labeled a future,
nondeployable audit. Sequence boundaries and unavailable final-row next events
are excluded appropriately.

| Panel | Scored destination label changes | Current event matches | Next event matches, oracle |
| --- | ---: | ---: | ---: |
| Fitted16 | 3,763 | 3,761 | 3,761 |
| Excluded16 | 3,842 | 3,841 | 3,841 |

All exceptions occur in class1: two fitted and one excluded. Thus the clock
condition remains nearly necessary in the actual scoring domain, but is not
exact and is not sufficient. Its replication across row-excluded groups is
descriptive corroboration with the fitting-component caveat above.

## What remains wrong in the strong model

Residuals are computed after clipping. Define symmetric modes
`e_sum=(e0+e1)/sqrt(2)` and `e_difference=(e0-e1)/sqrt(2)` and weight their
squared errors by the mean of the two absolute clipped target values. This
produces an additive *residual-energy diagnostic*, not an additive WP score.

Complete sum/difference residual energies are .2190/1.1734 fitted and
.2213/1.2987 excluded:84.3% and85.4% of this energy lies in the difference
mode. Class3 energies are .1813/2.0971 fitted and .1627/1.6741 excluded:
approximately 92.0% and91.1% lie in the difference mode. This identifies where
to inspect error, but the target variance is itself predominantly in this
mode; it does **not** prove that a sum/difference head improves representation.

The head increases pooled WP but worsens pooled absolute-target weighted MSE
for both targets in both panels. On fitted data parent weighted prediction
standard deviations are .980/1.054; head deviations rise to1.088/1.161 while
target deviations are1.286/1.303. The change involves prediction dispersion
and direction under the centered/clipped metric; it is not evidence of a
uniformly better conditional squared-error estimate. Post-hoc scaling is not
proposed from this fitted observation.

Age patterns require class conditioning. Pooled fitted WP falls from .8496
at age0 to .4728 at128+ rows since an observed event. But within class2 it is
.8781/.9234; class1 is .8412/.8010; class3 is .4108/.3325. The pooled monotone
pattern therefore mixes class composition and conditional difficulty. It
cannot justify a universal age decay or survival-rate decoder.

When the target pair is unchanged and no observed event occurs, the complete
predictor still updates. On scored class3 stable transitions, those updates
reduce weighted squared error on average for both targets (fitted reductions
.01394/.00822 per unit absolute target weight; excluded-row reductions are
.02119/.01049). Class0 t0 instead slightly
worsens, so even posterior-update benefit is not universal. Labels define this
stable-transition analysis only as an oracle diagnostic; they never gate input
or deployment.

Class3 weighted sign accuracy is about 82.3%/84.0% fitted and 81.8%/90.7% excluded,
yet wrong-sign predictions account for roughly 46%/49% and 50%/34% of its squared
error. Both direction and amplitude errors matter. Naive sign×magnitude
independence is not rehabilitated by this finding; its earlier matched cheap
control lost. A joint distribution can represent sign-dependent amplitude,
whereas an independent product assumes a factorization that need not hold.

## Method implication and next falsifier

Prioritize a small **joint conditional-distribution/residual head** or a
**transition/posterior-update branch** with matched capacity and the incumbent
complete head-training procedure. Train-defined support, causal latent regime,
and target-specific absolute-weight conditional decoding are the substantive
changes. An unconstrained sum/difference rotation is the mandatory capacity
control; rotations alone do not add information or representational power.

Any trained event branch must allow posterior updates inside observed quiet
runs, and beat equal-capacity/shuffled-event controls. Existing hard holding
is rejected. Additional component-excluded, seed-replicated comparisons are
needed before promotion; these fixed exposed panels can locate a problem but
cannot establish a 0.7000 expected-public solution.

Machine-readable per-sequence, target, class, age, event and sign diagnostics:
`residual_report.json` and `../strong_baseline_excluded/residual_report.json`.
Separate `event_age_residuals.png` figures accompany each panel.
