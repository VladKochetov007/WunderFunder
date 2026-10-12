# Bounded blend optimization primitive (R2)

This code-only addition provides small-array numerical functions for a later,
separately admitted endpoint comparison. It does not read prediction banks, bind
evaluation data, choose checkpoints, fit models, or establish an improved blend.
The synthetic tests do not use competition prediction arrays.

For one target with fixed weights `abs(clip(y, -2, 2))` and inactive prediction
clipping, the module computes weighted centered moments and the unclipped
correlation gradient. Solving `(C + ridge I) u = c` uses a linear solve, never a
matrix inverse. Positive ridge changes the problem; normalizing the positive
part of that direction is only an initializer. The averaged per-target
initializer is not an exact solution to the shared two-target objective.

`clip2_wp_surrogate_and_gradient` evaluates a continuous real-valued clip-2 WP
surrogate in float64 from float32 member arrays and provides its piecewise
analytic Jacobian. This is the SLSQP optimization objective. It is not the exact
official computation: the official `global_wp_clip2_v1` path casts its
prediction array to float32 before clipping and scoring. That float32 rounding
is discontinuous in blend coefficients, so this module does not claim or supply
an exact derivative of the official computational path.

`official_clip2_wp` forms a candidate mixture with float64 accumulation, casts
the resulting prediction array to float32, then calls the unchanged
`wnn.gmetric.global_wp`. It therefore gives the official metric for that
canonical candidate synthesis. The fit uses the continuous surrogate and its
gradient for at most three deterministic SLSQP starts (equal, current, and
regularized analytic); it reports both scores and ranks all feasible starts and
solver endpoints by the official float32 score. Each trajectory is limited to
50 iterations and constrains coefficients to the nonnegative sum-one simplex.
Optional shrinkage toward equal weights is a caller-fixed value, not a searched
parameter. This is a bounded local optimization, not a global-optimality claim.
Any delivered endpoint still needs scoring through its exact serving arithmetic.

The synthetic ULP test intentionally demonstrates a large proxy/scorer
difference: a float64 mixture varies enough to score 1.0 under the continuous
surrogate, while the canonical float32 cast collapses it to a constant and the
official score is 0.0. Other synthetic tests cover fixed-mask behavior,
unselected nonfinite values, selected nonfinite rejection, zero target weights,
and finite-difference checks away from clipping kinks.

R1's original note is retained at `NUMERICAL_PRIMITIVE_R1.md` with a supersession
notice; this R2 note is authoritative for the current API and numerical limits.

Historical context: the exposed F1 output study found residual-covariance
weights at `0.61119519` versus equal weights at `0.61119063` on its component
complement, a very small observed difference. This is not evidence of useful
generalization and does not reopen broad search. See
`research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/generated_r6/F1_BLEND_OUTPUT_EDA_R1.md`.

No competition weights, member selection, or real-data calibration have been
run with this primitive.

Focused verification command (using an existing environment only):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=120s \
PYTHONPATH=src scripts/limited .venv/bin/python -m pytest -q tests/test_blend_optimization.py
```
