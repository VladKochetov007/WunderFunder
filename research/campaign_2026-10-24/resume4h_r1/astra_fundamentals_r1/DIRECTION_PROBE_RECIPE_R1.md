# Proposed fixed-prediction direction probe

Status: frozen analysis suggestion for root/independent grader review; not admitted or executed by this agent. Zero new model fits, inference passes or public queries. Objective: distinguish useful shared correction directions, parent calibration directions and relationship-specific directions in the already-saved predictions. This diagnoses fixed endpoints; it does not identify a training cause.

## Scope and resources

Use the exact F0 parent and five unique A/B/C predictions from `../feasibility_scout/F0_EXPLORATORY_RECIPE_R2.json` and the same source-qualified 468 sequences, 141 components, 808,863 rows, original mask and target archive. Preserve producer pins and existing ancestry checks. F1 A is a separately labeled extension only if its independent source/ancestry preflight passes; never delay F0 waiting for it. Folds are exposed; this probe is selected after the raw and shrink grades.

One worker, one CPU, hard 400,000,000 bytes, zero swap, no Torch, physical-disk scratch. Target <900 CPU seconds; terminate at 900 CPU seconds or the active campaign collection deadline, whichever comes first. Existing 33/49-second scoring receipts suggest feasibility, but do not prove this implementation's cost. Root must admit against aggregate memory. Stream source-qualified sequence/chunks, memory-map existing uncompressed arrays where supported; do not decompress a whole ancestor cache or rebuild features. Reuse the validated target-reader; if it cannot fit safely, stop with an execution finding. Per-component moment tables are tiny. No coefficient sweep or top-candidate selection.

## Frozen estimands

For one target, write raw parent prediction `p`, raw candidate `q_j`, correction `d_j=q_j-p`, clipped label `y`, official weight `w=abs(y)`, and `c=clip(p,-2,2)`. All means, covariances and standard deviations below use the fixed selected rows and weights. Keep each target separate, then average its score derivatives, matching the official score. Compute moments in float64; construct deployed alpha=0.1 predictions using the already-frozen float32 arithmetic.

Let `z_j` be the right directional derivative of `clip(p+alpha*d_j)` at zero: `d_j` for `-2<p<2`, zero for `abs(p)>2`, `min(d_j,0)` at `p=2`, and `max(d_j,0)` at `p=-2`. Then

`D_j = Cov_w(y,z_j)/(sd_w(y)*sd_w(c)) - rho_w(y,c)*Cov_w(c,z_j)/Var_w(c)`.

Report degeneracy explicitly when an official variance is zero. Also calculate the same derivative at the one already-fixed alpha=0.1, using `p+0.1*d_j` as the base and `d_j` as its direction. These derivatives are mathematical diagnostics in real arithmetic; finite float32 differencing is not required to equal them bit-for-bit.

Primary outputs, for every endpoint and both targets:

1. `D_at_0`, `D_at_0.1`, the exact existing alpha=0.1 score delta, and raw alpha=1 delta. Reproducing known scores is a reader/alignment check; do not fit a coefficient.
2. Parent saturation and exact-boundary counts and official-weight mass; alpha=0.1 and alpha=1 clipping-entry/exit counts and mass. Keep unweighted counts separate from weighted mass.
3. Calibration-adjusted and shared-control-adjusted direction effects, as defined below. These are descriptive moment decompositions, not deployable calibrators or fitted models.

## Controls and projection

The projection is calculated at alpha=0 only, with the shared raw parent p as the base for every candidate and A. Alpha=0.1 requires derivatives and clipping diagnostics, not a second projection. Compute parent-calibration tangent directions for the fixed raw prediction changes `+1`, `-1`, `+p`, and `-p`, each through the same right clipping derivative. Away from exact clipping boundaries, opposite pairs are redundant. Keeping both signs handles the one-sided kink without assuming exact-boundary mass is zero. Center these directions under `w`; use their weighted span `N_cal` as the nuisance basis.

For each `z_j`, report the derivative functional applied to (a) its projection onto `N_cal`, and (b) the remaining direction. The two contributions must sum numerically to `D_j`. For B/C additionally extend the basis with the centered A direction to form `N_cal+A`, and report the remaining score-direction contribution. A common-A direction itself need not be calibration.

The weighted Gram matrix and cross moments suffice. Drop and report zero-variance nuisance columns; standardize positive-variance columns to unit weighted variance, then eigendecompose their weighted correlation Gram in float64. Retain eigenvalues strictly above the largest eigenvalue times 1e-10; this is an eigenvalue threshold on the standardized Gram, not a singular-value threshold on the design. Report the eigenvalue spectrum and retained rank; materially negative eigenvalues are a numerical finding, not silently discarded evidence. Do not interpret an ill-conditioned residual as a precise mechanism estimate. Coefficients use evaluation weights derived from labels; they are prohibited from being exported as a serving rule. The residual of a tangent projection need not itself correspond to a realizable one-sided perturbation at every clipping boundary, so it is a geometric allocation of the derivative, not a counterfactual score.

Save additive per-component weighted first/second moments for labels, base predictions, candidate tangents and nuisance directions. A fixed 2,000-draw paired component bootstrap with seed 20261024 may recompute the complete moment calculation and projection for each draw. Report numerical-rank changes. This gives conditional evaluation uncertainty only; it does not include training, fold, exposure, selection or transfer uncertainty. Include all declared endpoints, not a winning interval.

## Decision-changing interpretations

| Observation | What it changes | What it does not establish |
| --- | --- | --- |
| Positive derivative at zero, weaker/negative derivative at 0.1 | The direction/scale distinction is material; prioritize a training-scale or regularization question over assuming no signal. | Optimizer overshoot, teacher optimism or a deployable optimal coefficient. |
| Most common A direction lies in calibration span | A separate calibration comparison can be cheaper than more online feature fits. | That calibration caused training failure, or that the evaluation-derived projection is legal serving. |
| A retains a useful direction outside calibration | Generic trained readout or input-state information remains plausible; run the missing zero-stat control. | Online-statistic value, since A also adds head capacity. |
| B/C retain a useful direction beyond calibration+A | Relationship-specific follow-up has a concrete direction to test across cells. | A causal slope mechanism or public transfer. |
| B/C add little estimated direction beyond A | Reduce priority of repeating this exact slope recipe; use its uncertainty and cost. | Equivalence or universal rejection of online relationships. |
| F1 A disagrees with F0 | Investigate parent/cell dependence before treating one common correction as portable. | Which of fold, seed, ancestry or sample variation caused disagreement. |

Completion artifacts: source/array/membership pins, one machine-readable moment/results file, execution receipt with CPU/RSS/swap, all estimates and conditional intervals, a brief independent formula/alignment review. Root owns admission and integration; an independent grader owns payload execution. No launch authority is conveyed by this file.
