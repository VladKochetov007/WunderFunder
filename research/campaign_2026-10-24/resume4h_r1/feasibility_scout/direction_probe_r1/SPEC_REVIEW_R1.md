---
kind: review
status: complete
created_utc: 2026-10-24T12:57:19Z
updated_utc: 2026-10-24T12:57:19Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/astra_fundamentals_r1/DIRECTION_PROBE_RECIPE_R1.md
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/F0_EXPLORATORY_RECIPE_R2.json
  - src/wnn/gmetric.py
---

# Direction probe formula review

## Scope and disposition

Reviewed the frozen proposal and official scorer source only. No prediction or target payloads, native code, or model imports were accessed. The proposed derivative and signed one-sided clipping tangents are mathematically correct **for the unclamped weighted Pearson branch when its variances are nondegenerate**. Before admission, the machine recipe should encode the official scorer's degeneracy and final-correlation clip behavior, preserve the float32/real-arithmetic distinction, and retain full pairwise moments for bootstrap reconstruction. This is a useful fixed-endpoint geometry diagnostic; it cannot identify a training cause or a deployable calibration.

## Formula and estimand checks

For each target, with fixed `w=abs(clip(y,-2,2))`, `c=clip(p,-2,2)`, and tangent `z`, differentiating `rho=Cov_w(y,c)/(sd_w(y) sd_w(c))` gives

`D(z)=Cov_w(y,z)/(sd_w(y)sd_w(c)) - rho*Cov_w(c,z)/Var_w(c)`.

Thus the recipe's expression is right when all weighted moments use the same normalization, the target and weights stay fixed, and the official output is not on a degenerate or clipped-correlation branch. Compute each target derivative independently and report the official overall derivative as `(D_t0 + D_t1)/2`; do not pool targets or weights across t0/t1. The projection contribution identity is linear, so `D(Pz)+D(z-Pz)=D(z)` should hold up to numeric tolerance. The real-valued path derivative is not the same object as the exact saved float32 alpha-0.1 prediction: grade the latter with the frozen float32 construction and compare its saved score delta to the existing grade. Do not expect finite differences of float32 endpoints to equal the analytic derivative bit-for-bit.

`src/wnn/gmetric.py` sets `_EPS=1e-8`: it returns zero when total weight, target SD, or prediction SD is below the official threshold, and clips the final correlation to [-1,1]. The recipe currently says to report variance degeneracy, but the implementation contract should match all three zero-score branches (`W < EPS`, `sd_y <= EPS`, `sd_c <= EPS`) and record distance to those thresholds. Report the analytic quotient only on its regular branch. At an exact threshold, classify the one-sided gate behavior explicitly; do not substitute a finite quotient formula. Likewise apply the one-sided derivative of the final `clip(rho,-1,1)` if the base correlation is exactly at a cap. An inactive zero-score branch has zero derivative while the gate remains inactive.

The clipping tangent cases in the proposal are correct: strict interior gives `d`; strict saturation gives zero; at `p=2` use `min(d,0)` and at `p=-2` use `max(d,0)`. Apply those cases at both the alpha-0 base `p` and real-path alpha-0.1 base `p+0.1d`. Count strict saturation, exact-boundary mass, and transitions separately, with weighted mass distinct from row counts. Signed calibration tangents are a one-sided tangent **span**, not a feasible calibration cone. Consequently projection shares and derivative contributions are geometric decompositions, not counterfactual scores. Include weighted norm share `||Pz||²_w/||z||²_w` if interpreting that a direction is “mostly in” a nuisance span; a derivative split alone does not establish geometric overlap.

## Required sanity checks before any real-data execution

1. On small synthetic weighted arrays with both targets nondegenerate and predictions away from clip/correlation boundaries, compare the analytic derivative with float64 one-sided finite differences at alpha 0 and alpha 0.1 for decreasing `h`. Separate tests should place raw `p` exactly at ±2 with both signs of `d`, strictly beyond each bound, and within the interior; finite differences must match the specified right tangents.
2. Check the two-target average against the derivative of the averaged official score; check that alpha 0 reproduces the parent and exact alpha 0.1/1 scores reproduce existing fixed endpoint grades using their prescribed arithmetic. Test zero-weight, `sd_y <= 1e-8`, `sd_c <= 1e-8`, exact-gate, and final `rho=±1` branches against the canonical scorer.
3. For each projection, verify residual weighted orthogonality to the retained eigenspace, `D(Pz)+D(z-Pz)=D(z)`, and `rank(N_cal+A) >= rank(N_cal)`. A zero-variance column must be dropped; report the complete standardized-Gram spectrum, strict eigenvalue cutoff, and any negative eigenvalues. Include projected norm fractions so “overlap” is not inferred from contribution signs.
4. For each target and component, save additive sufficient statistics for the complete vector `x=(y,c,z_1…z_m,n_1…n_k)`: total weight, every weighted first moment, and the full weighted cross-product matrix `sum(w*x_i*x_j)`. Marginal second moments alone cannot reconstruct cross-covariances or Gram matrices. On a tiny synthetic fixture, compare resampled-component aggregation from these tables with direct row-level recomputation, including rank changes.

F0 contains five unique saved directions because its A lag-1 and lag-16 predictions are identical; count that endpoint once. Keep F1 A separate under its own parent, source-qualified membership, target weighting, and bootstrap. The F0/F1 evaluations are exposed and the probe follows prior grades, so intervals are conditional diagnostics and not fresh confirmation. A positive projected or residual derivative is not itself an observed score gain or a causal mechanism result.
