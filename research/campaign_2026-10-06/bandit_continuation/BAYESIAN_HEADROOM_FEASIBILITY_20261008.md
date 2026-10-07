# Bayesian headroom estimates: feasibility and research use

## Decision

The user's proposal is meaningful if separated into (1) conditional forecasts of
additional training/scaling gains and (2) the theoretical predictability ceiling
of the supplied causal observations. These are different quantities. No numerical
posterior or nontrivial data-wide upper bound has been estimated here. The only
unconditional score upper bound currently established is 1.0. This is not evidence
that the attainable score is near 1.0, nor evidence against the >0.700 objective.

Use forecasts for bounded compute allocation, never as an automatic arm kill.
Current heterogeneous recipes and exposed validation results cannot identify a
universal ceiling. A plateau of our optimizer or representation is not a noise
floor of the data. The existing public scores are not a matched scaling ladder.

## What the official metric implies

Inspected source: `src/wnn/gmetric.py`, metric `global_wp_clip2_v1`. On the fixed
selected rows, each target and prediction is clipped to [-2,2]; each target has
weight equal to its own clipped absolute magnitude. The final score averages two
pooled weighted correlations. This is not ordinary unweighted squared error.

The following is a population derivation, not an empirical result. For one target
let Z=clip(Y,-2,2), w=|Z| and H be all permitted causal information available at
prediction time. Work under the selected-row distribution. Define a tilted
probability Q by dQ=w dP/E_P[w]. Its conditional mean is

    m(H) = E_P[w Z | H] / E_P[w | H].

Where the denominator is zero, the prediction is immaterial to the metric. Under
nondegenerate finite variances, the maximal population correlation over all
measurable causal predictors is

    rho_star = sqrt(Var_Q(m(H)) / Var_Q(Z))
             = sqrt(1 - E_Q[Var_Q(Z | H)] / Var_Q(Z)).

This follows from conditional expectation and Cauchy–Schwarz: covariance of Z
with any function of H equals covariance of m(H) with that function. The optimum
m lies within [-2,2], so the prediction clipping constraint does not invalidate
this population optimum. Apply the derivation separately to both targets and
average their optima; serving/model restrictions can reduce what is attainable.
The expression assumes a fixed stationary population and causal information set;
it does not guarantee transfer to a shifted public population.

Important consequence: an ordinary conditional mean or homoscedastic MSE noise
estimate is not automatically the optimal predictor or ceiling for this metric.
The label-dependent weighting changes the relevant distribution. A model's
remaining residual includes approximation, optimization and estimation errors,
so treating all of it as irreducible noise creates an unjustified upper bound.

## Why a number is not identifiable from our present evidence

We do not have independent repeated future outcomes for exactly the same full
causal history. Nearby histories can still contain relevant different information;
their label variation is not a demonstrated noise floor. Duplicate rows within
one trajectory are not independent repeats. Ensemble disagreement measures model
variation, not all irreducible uncertainty. High-dimensional finite observations
admit very different conditional laws unless smoothness/noise assumptions are
introduced. Bayesian inference makes those assumptions explicit; it cannot remove
this ambiguity.

Likewise, fitting a ceiling to models with different teachers, data membership,
fold exposure, precision or blending conflates mechanisms. An empirical model
using future information can be a diagnostic oracle, but its score is not a
certified upper bound and cannot be submitted. Such an oracle must remain isolated
from causal fitting, model selection claims and serving.

## A useful bounded analysis after matched runs land

1. Extract epoch-level scores and completed optimizer updates from the same
   frozen recipe, split and metric. Keep all epochs, not only selected winners.
   Separately assemble matched data-size and model-size ladders; vary one resource
   dimension at a time before interpreting its scaling effect.
2. Fit several modest Bayesian curve families: saturating power law, exponential
   saturation and a broken/nonmonotonic alternative. Separate training-seed
   variation from chain-level evaluation dependence. Account for correlation among
   epochs scored on the same chains; epochs are not independent replications.
3. Backtest by hiding later observed points. Compare posterior coverage and
   predictive error against the simple last-observation forecast. Report prior
   sensitivity, especially for an asymptote unsupported by observed curvature.
   If forecasts fail this check, do not use their attractive asymptotes.
4. Prefer finite questions: probability of a meaningful gain from another 2x
   training budget, likely gain per GPU-hour, or probability a fixed endpoint
   improves a matched control. Report these as recipe/family-conditional forecasts,
   not public-score probabilities or the ceiling of all future architectures.
5. Preserve an exploration reserve and full-budget lane. New features, objectives,
   memory mechanisms, data treatments and combinations can change the curve.
   A posterior based on an old family is not evidence to reject a new one.

No dedicated GPU fits are added merely to make a ceiling plot now. Reuse matched
epoch logs as they land, then admit a small scaling ladder when it also tests a
scientific direction. This protects experiment throughput. Numerical estimates
remain pending; no invented prior mean or score ceiling is entered in the tree.

## Verified primary literature

- [Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted
  Networks, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract.html):
  posterior predictions of later training performance are a real, evaluated
  research method. Its reported benchmark speedups are not measured speedups in
  our competition. LC-PFN's synthetic curve prior needs suitability checks here.
- [Revisiting Neural Scaling Laws, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/8c22e5e918198702765ecff4b20d0a90-Paper-Conference.pdf):
  validates curve models using extrapolation error rather than trusting an
  in-range fit. This motivates the hidden-later-point backtest.
- [Broken Neural Scaling Laws, ICLR 2023](https://arxiv.org/abs/2210.14891):
  documents richer transitions and nonmonotonic scaling. One smooth saturating
  fit is not a universal description of architecture changes or delayed gains.
- [Automatic Feasibility Study via Data Quality Analysis for ML: A Case-Study
  on Label Noise](https://arxiv.org/abs/2010.08410):
  Bayes-error estimation has practical literature, but this classification-focused
  case study does not establish an estimator for our causal weighted regression.

Sources inspected 2026-10-08. The metric-specific derivation above is our reasoning,
not a claim that these papers prove a ceiling for this competition.
