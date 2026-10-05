# Independent Freebuff DeepSeek V4.1 Flash — low effort

# Final review — GLM critique, four component interventions, bounded study

## 1. Independent critique of the earlier GLM proposals

**Hazard-mixture event model — retract.** The new panel evidence is decisive: freezing predictions until observed `p0/p11` events moved WP 0.7744 → 0.6929, and row-excluded 0.7428 → 0.6665. The freeze-between-events / hold-until-event premise is falsified, and with **no identified physical price decoder** the event labels are circular (events defined from target Δ, then used to predict target Δ). Its falsifier (a) — Δt structure — would have caught this cheaply; I should have run it first.

**Joint bivariate Gaussian + monotone-spline head.** Unidentifiable at this effect size: seed/partition spread (~0.002–0.004) exceeds the head effects it claims (+0.0008 to +0.0019), and `ρ̂` will be fit almost entirely to the sparse t1. More fundamentally, a joint likelihood reaches the metric only through the **pooled covariance**; per-target means do not enter a globally pooled centered correlation the way the proposal implied. "Strong t1 regularizes t0" needs disjoint panels before it is anything but a fitting artifact.

**Heteroscedastic `é(x)`-weighted head.** Two defects. `ê` is self-consistently trained with no constraint, so it can collapse to a constant or absorb trunk signal (zero-sum with `m`). Worse, pointwise weighted MSE is only a proxy: weighted Pearson is a **globally pooled, centered, normalized** statistic, so a loss that reweights rows cannot by itself reproduce the metric's nonlocal centering and scaling. Keep "estimate `|y|`"; drop "loss weighting approximates the metric".

**Common flaw.** All three assumed per-class or per-event structure is exploitable. The metric is not a sum of class correlations, and class3 ≈ 91% means class decompositions largely restate the global statistic rather than adding independent signal. Also, the diagnostic panels still overlap historic fitting, so all quoted WP deltas are relative and likely optimistic.

## 2. Metric-relevant mean vs ordinary likelihood mean

The scorer minimizes a **pooled, `|y|`-weighted, globally centered** squared-error-like objective after clip. The metric-relevant target is therefore the `|clip(y)|`-weighted conditional mean evaluated over the pooled row population — *after* global centering and scale normalization. This differs from the ordinary likelihood mean in two ways: (i) rows are weighted by predicted `|y|`, so residuals in high-|y| regions dominate; (ii) centering is global, not per-sequence or per-class, so any trainable normalization must be pooled at training time, not applied post-hoc. Any component that claims the metric must show gains on the pooled statistic, not on local class fits.

## 3. Four component interventions

**A. Multi-timescale linear state-space memory (SSM) replacing the recurrent credit.**
*Mechanism:* diagonal linear SSM (S4D/Kalman form), decays initialized to span the incumbent EMA schedule; state is a causal multi-timescale weighted sum of book/trade moments; pooled-centered linear head per target. O(1) sequential inference fits 1 CPU / 20 MB.
*Matched control:* identical features+head with (i) instantaneous MLP (no memory) and (ii) incumbent-class GRU at equal state budget/parameter count.
*Cheap falsifier:* learned decay spectrum is unimodal/degenerate (no multi-timescale content) **or** instantaneous MLP matches pooled WP within the seed CI ⇒ memory claim dead.

**B. Content-gated non-Transformer attention on the difference pathway.**
*Mechanism:* exploit the measured 84–85% target-difference residual *only if* it is predictable: predict shared level with A, then route the difference through selective (input-gated) linear attention / selective scan — O(N), no softmax — so the model can retrieve the sparse past rows predicting a target difference.
*Matched control:* (i) same module with the gate clamped content-independent (constant decay); (ii) soft weighted vs hard top-k retrieval.
*Cheap falsifier:* zeroing the difference pathway leaves pooled WP unchanged within seed CI ⇒ "residual dominance" is descriptive variance, not exploitable structure; stop.

**C. Small KAN-inspired spline edges, restricted to proven heteroscedastic channels, plus `ê` head.**
*Mechanism:* degree-3 B-splines on a small grid (≈8) only on per-level book features, activity and volume-like columns where `|y|` predictability is already evidenced (volume28 +0.00096, W384 +0.0020); a separate constrained `ê(x)` head estimates expected `|clip(y)|` to steer gradient weight. Keep edges few (20 MB cap); this is a differentiable parametric family, not post-hoc calibration.
*Matched control:* identical topology with linear edges; KAN with identity grid; and no `ê` head.
*Cheap falsifier:* LR test — `|y|` predictable from these features at all (flat `ê` ⇒ metric-emulation premise void); spline-edge vs linear-edge pooled-WP delta inside the paired bootstrap CI ⇒ component unjustified.

**D. Training-time pooled centering/normalization module.**
*Mechanism:* a first-class global module that estimates pooled weighted mean and scale over the whole training batch and applies the metric-consistent centering inside the objective — the operational form of §2. Distinct from dead post-hoc rescaling/soup (≤6e-05) because it changes gradients during training, and distinct from H-073 normalizer alignment because it targets the pooled scorer statistic itself.
*Matched control:* per-sequence normalization; and post-hoc pooled normalization at eval only.
*Cheap falsifier:* gain disappears under paired bootstrap on **fully disjoint** panels, or appears only on panels overlapping historic fitting ⇒ contamination.

## 4. Assumption critique

(1) *Event observability* — falsified by the p0/p11 freeze loss; no physical decoder exists. (2) *Per-class transferability* — invalid under global pooling; class3 ≈ 91% makes per-class controls low-power. (3) *Difference-residual dominance ⇒ predictability* — dominance is a variance fact; sale comes only from B's ablation. (4) *Weighted loss ≈ metric* — only a proxy under global centering; §2 governs. (5) *Linear SSM sufficiency* — linear state spaces cannot express content-dependent gating, hence B is not redundant with A. (6) *Spline identifiability* — low-order, regularized grids required; check null-space before attributing gain. (7) *Diagnostic panels* — historic-fitting overlap inflates all measured WP; never compare across panels. (8) *Seed variance* — promotion requires a new seed/partition, not a second fold.

## 5. Finite study (no 0.7 promise)

Bounded matrix: 4 interventions × 2 matched controls = 8 configs, one epoch/seed each on a single fixed, disjoint 1k-sequence panel; screen on pooled WP with the official scorer; promote to a new seed/partition only if the paired component bootstrap interval excludes zero and the gain exceeds the observed seed spread. Cap: ≤ 8 GPU-h training, inference checked against 60 min / 16 GB / 20 MB. Kill rule pre-registered: any intervention failing its own falsifier above is dropped without a second configuration. **Expected outcome is bounded resolution of which components carry real pooled signal — not a public-score target.**
