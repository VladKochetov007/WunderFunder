# Bounded blend optimization primitive (R1)

> Superseded by [R2](NUMERICAL_PRIMITIVE_R2.md): R1 described the continuous
> float64 clip-2 surrogate as the official computation. R2 corrects that
> precision qualification and scores candidates through the official float32
> metric path. This R1 note is retained as historical context.

This code-only addition provides small-array numerical functions for a later,
separately admitted endpoint comparison. It does not read model banks, bind
evaluation data, choose checkpoints, fit models, or establish an improved blend.
The synthetic tests do not use competition prediction arrays.

`src/wnn/blend_optimization.py` has two distinct methods. For one target with
fixed weights `abs(clip(y, -2, 2))` and inactive prediction clipping, it computes
weighted centered moments and the unclipped correlation gradient. Solving
`(C + ridge I) u = c` uses a linear solve, never a matrix inverse. Positive ridge
changes the problem; normalizing the positive part of that direction is only an
initializer. The averaged per-target initializer likewise is not an exact
solution to the shared two-target objective.

The shared fit evaluates the official fixed-mask, clip-2, target-magnitude
weighted Pearson mean for both targets. It supplies the piecewise analytic
Jacobian, with zero derivative at exact clipping boundaries, to at most three
deterministic SLSQP starts: equal, caller-provided current weights, and the
regularized analytic initializer. Coefficients are constrained nonnegative and
sum to one. Each trajectory is limited to 50 iterations. Optional shrinkage is
a caller-fixed value toward equal weights; the function does not search it.
SLSQP is a local optimizer, not a global-optimality claim.

The objective accumulates the continuous coefficient mixture in float64 from
float32 member predictions. The official scorer casts delivered prediction
arrays to float32. Consequently, the test checks score parity within a tight
floating-point tolerance; every eventual exported candidate must still be
re-scored by the unchanged official scorer. No competition weights, member
selection, or real-data calibration have been run with this primitive.

Historical context: the exposed F1 output study found residual-covariance
weights at `0.61119519` versus equal weights at `0.61119063` on its component
complement, a very small observed difference. That is not evidence of useful
generalization and is why this work adds a tested primitive rather than claiming
a new result or reopening broad search. See
`research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1/generated_r6/F1_BLEND_OUTPUT_EDA_R1.md`.

Focused verification command (using an existing environment only):

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=650000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python -m pytest -q tests/test_blend_optimization.py
```
