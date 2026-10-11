# Optimization of complete submission blends

Root planning, 2026-10-12 Europe/Kyiv. This responds to the user's request for
analytical, constrained, quasi-Newton and random-search methods. It extends
ADR0033; it does not enable HPO, new spending or an unreviewed GPU fit.

## Decision and existing evidence

Optimize the delivered endpoint: selected independent models, fixed coefficients,
transport/runtime precision and causal implementation. The objective is official
`global_wp_clip2_v1` on source-qualified evaluation groups. Size, one-CPU execution
time and valid deterministic reset behavior are constraints. Use actual ZIPs and
callback measurements rather than presumed bit-width speedups.

The existing exposed-F1 selector already compared equal family weights,
calibration-only inverse-output-SD weights, 80%-shrunk residual covariance
weights and overall output scaling. On its component complement, covariance
weights gave .6111952 versus .6111906 for equal weights; scaling changed WP by
1.44e-8. This is little observed reward for weight fitting, not a proof that
every constrained solver or new family pool is useless. The three independent
teacher seeds were more consequential: full exposed-F1 trio .6725416 versus
mean individual .6706805. Adding the narrow student at half family weight gave
.6726401, only +.0000985 over that trio. These measurements condition on fixed
predictions; they do not quantify public transfer or fresh cross-validation.

Prioritize exact native-precision endpoint confirmation and missing cross-fold
families over many more searches on this same fold. Keep equal weights as a
frozen control. All inspected folds remain exposed.

## Analytical weighted-correlation solution

For **one target, no active prediction clipping**, take the official fixed
mask and fixed weights d_i=abs(clip(y_i,-2,2)). Center each candidate prediction
column and the clipped target using these weights. Normalize sums by total
weight; let C be the prediction covariance matrix, c the prediction/target
covariance vector and v_y the target variance. For coefficients a,

    r(a) = (c^T a) / sqrt(v_y * a^T C a)

For positive numerator and nondegenerate covariance, an unconstrained maximum
has direction C^{-1}c; solve the linear system, do not explicitly invert C.
Highly correlated models make C ill-conditioned. Ridge/shrinkage is a declared
regularization choice, not the exact unregularized optimum, and can improve
stability at the cost of calibration fit.

With nonnegative coefficients, the single-target, unclipped problem can instead
be expressed as an SOCP: maximize c^T a subject to ||L a||_2 <= 1 and a >= 0,
where L^T L=C. Normalize the nonzero solution to sum one afterwards. This uses
correlation's positive-scale invariance and presumes a positive achievable
covariance. Degenerate zero-variance directions require removal or an explicit
regularizer. Family ratio constraints that are homogeneous can be included;
ordinary simplex-normalized caps need a careful transformed formulation.

This is our derivation, using standard cone-program geometry described in
[Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/). It is **not** a
claim that our exact two-target clipped competition objective is convex.

For the smooth unclipped case the analytic gradient is

    grad r = c / sqrt(v_y * a^T C a) - r * (C a) / (a^T C a).

Each target has its own C, c and label-magnitude weights. The official score is
the mean of two correlations. Shared blend coefficients therefore generally
need numerical optimization; averaging two analytic coefficient vectors is
not an exact solution. Target-specific coefficients are a separate serving
arm and require an independently reviewed consumer implementation.

## Exact constrained numerical optimization

Use a small bounded SLSQP problem for shared coefficients:

    minimize -official_WP(a) + lambda * ||a-a_equal||_2^2
    subject to a >= 0, sum(a)=1, declared family caps if applicable.

SLSQP accepts derivatives, bounds and equality/inequality constraints; its
termination and constraint residuals must be recorded alongside the result.
[Primary SciPy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

Compute pooled weighted moments in streams. For the actual clipped prediction
p_i=clip(sum_j a_j p_ij,-2,2), the prediction derivative is the original column
p_ij on unsaturated rows and zero on saturated rows, except at boundary
nondifferentiabilities. Labels, weights, mask and official clipping remain fixed.
The centered weighted-correlation gradient follows by differentiating its
covariance and variance. Validate analytic derivatives on calibration-only
nonboundary fixtures and re-score final candidates through the unchanged
official accumulator. A differentiable surrogate is allowed for optimization
only when explicitly labeled and the delivered endpoint is graded officially.

Use equal and the regularized analytical solution as at most two initializations;
bound iterations and complete attempted-set exposure. Solver success establishes
numerical convergence at most, not global optimality or generalization. Flat
objective differences around floating-point precision do not justify new ZIPs.

L-BFGS can optimize a smooth reparameterization, but softmax excludes exact zero
weights and cannot directly express every family cap. It is a secondary choice
here; full Newton/Hessians are unnecessary for a handful of coefficients. Do
not run continuous optimizers over discrete precision/model-family choices.

## Discrete model and precision allocation

Enumerate a small predeclared candidate set or use a beam/greedy search with
explicit pair interactions. A weak standalone model can improve a strong blend;
screen its conditional contribution before dropping it. After each discrete
choice, fit only the bounded coefficient procedure on calibration components.

For binary inclusion z_j and precision choice q_j, measured constraints have the
form ZIP(z,q)<=20,000,000 bytes and wall(z,q)<=3,600 seconds on the competition
CPU. This is a mixed discrete/continuous problem. ZIP compression and callback
costs are not exactly additive; a mixed-integer or knapsack surrogate can propose
choices, but actual assembled ZIP and exact callback gates decide feasibility.
Use a measured runtime margin; local microseconds do not certify competition
hardware wall time.

Random search is useful for a small pool of family/subset/precision combinations
when interactions defeat a simple greedy policy. It is low priority for three
nearly collinear teacher weights. Predeclare a bounded number of choices and
seed, retain all results, and do not restart the disabled HPO infrastructure.
Evolutionary/Bayesian methods add little until a larger useful search space and
reliable endpoint cost/quality observations exist.

## Robustness and release

Use complete source-qualified components for calibration and a separate fixed
component complement for evaluation. This is an exposed development diagnostic,
not fresh validation. Compare paired component effects and existing cross-fold
recipe evidence. Prefer stable nonnegative weights and shrinkage; avoid a large
signed cancellation portfolio. Report training-seed variation separately from
fixed-prediction resampling. A zero-crossing interval is not a universal kill
switch; allocate uncertain follow-ups by information value and opportunity cost.

Current next useful experiments are the native F1 teacher-pair/trio precision
panel and clean F0 teacher/student pipeline. A new constrained-weight candidate
must beat or robustly tie the complete equal-weight endpoint after precision
and runtime gates. Do not transform a local gain into an expected public number.

## Existing QAT code and its limit

`src/wnn/training_qat.py` already provides weight-only recurrent or recurrent/head
Q8 fake quantization with identity STE, detached row scales and CPU canonical
rounding cached once per latent update. It keeps activations/state FP32 and
exports represented weights plus encoded rows. It does not implement learned
Q5–Q7 scales or the deployed dynamic activation-Q8/GELU/normalization composition.
The head preset quantizes separate column blocks; the serving consumer must
match that representation. Reusing this helper is an implementation starting
point, not proof of faithful Q8VNNI QAT. Preserve codes/scales in an audited
consumer before promising fine-tuning benefits or admitting matched GPU tests.
