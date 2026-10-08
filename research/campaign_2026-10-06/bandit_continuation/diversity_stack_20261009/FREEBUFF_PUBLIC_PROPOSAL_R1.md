# Wunder Connectome Challenge 2 — Independent Experiment Program Proposal

**Status:** bounded ideation; no fits launched, no files read, no tools used. All facts below come from the brief; all model-level claims are labeled proposed inference. Recalled-from-memory citations are labeled **[unverified]**. The Markowitz prior is used only as a structural lens, with its disanalogy stated.

---

## 0. Standing constraints (taken as fixed from the brief)

- **Causality:** predictions use only current/past features of the current sequence; no labels at inference; all recurrent state resets on `sequence_id` change (source-qualified, e.g. `train#042`, `valid#017`, to prevent namespace collision).
- **Disjointness:** evaluated groups/components, all teachers, all initializers, and all target ancestry must be disjoint from the data used to fit any selector/stacker. Cross-fit everything that touches labels at selection time.
- **Folds:** previously examined folds are burned. New fold identities must be freshly drawn (recorded seed), GroupKFold over source-qualified sequence IDs, complete 20,000-row sequences only, warmup rows 0–98 excluded from loss but **retained in screens** (they feed state).
- **Metric:** `global_wp_clip2_v1` — fixed ±2 clip on predictions and labels, original clipped-label magnitude weights, pooled weighted correlation per target. Training-side clipping/loss/target is a treatment, not an assumption.
- **Serving:** deterministic Python 3.11, 1 CPU, 16 GB RAM, ≤20 MB ZIP, ≤60 min measured end-to-end. ~36 min is one measured calibration point, not a budget.
- **Compute:** 1 local GPU (≤5.x GB VRAM), 5 CPU/20 GB aggregate, Kaggle T4/TPU only with device parity validated, Vast within existing credit. No new spend. Warm cohorts verify once, parse/mmap once, keep predictions/checkpoints on SSD.

---

## Card 1 — Heterogeneous teacher panel: a factorial core, not a pile of models

**Mechanism.** Diversity that helps blending comes from *decorrelated errors*, which are produced most reliably by varying the loss surface, the optimizer geometry, the activation class, and the effective history length — not by re-seeding one architecture. Build a small factorial grid and keep only cells whose OOF errors are sufficiently decorrelated from the existing panel.

**Algorithm (one teacher cell).** GRU with hidden size h, input = 112 features (+ optional per-sequence normalized step encoding), trained on complete sequences with truncated BPTT of length L, state zeroed at sequence start (rows 0–98 consume but emit no loss). Factorial axes:

| Axis | Levels |
|---|---|
| Optimizer | Muon (hidden matrices), AdamW (all params); control: SGD+momentum on one cell [unverified: Muon's orthogonalized-update geometry produces error decorrelation vs AdamW] |
| Activation | tanh (GRU default), GELU-in-output variant, identity+layernorm variant |
| Loss | Huber(δ ∈ {0.1, 0.5}) on clipped labels; MSE on raw labels (clip only at serve); pairwise weighted-correlation loss |
| History | truncated BPTT L ∈ {256, 2048}; one full-sequence (L = 20000, gradient checkpointed) cell only if VRAM allows |
| Target treatment | train on raw y vs train on clip(y, ±2) vs train on rank-gauss(y) [unverified: rank-gauss stabilizes heavy tails] |

**Information at fit time:** labels, full histories, fold assignment. **At serve time:** features + step + output flag only.

**Ancestry / cross-fit construction.** Every teacher is trained on training-fold sequences only. Its OOF predictions on validation folds are produced by a model that never saw those sequences' labels. Fold identities drawn once, recorded, and shared by the whole program so any two artifacts' ancestry is checkable from a manifest.

**Candidate vs matched control.** Candidate: the factorial cell. Control: the same cell with the axis of interest reverted to the existing panel's default (e.g. AdamW→Muon cell vs AdamW baseline), identical seed budget (3 seeds), identical fold assignment.

**Predicted observable.** Pairwise Spearman correlation of OOF residuals between the new cell and each existing teacher < 0.9 while its standalone weighted corr is within ~0.02 of the panel's median. A cell that is great but correlated ≥0.95 with an existing teacher is not worth serving cost.

**Cheapest informative test.** One Muon-vs-AdamW matched pair at L=256, both targets, 1 seed each, single fold pair. ~2 GPU-hours total. If residual correlation between the pair > 0.98, the optimizer axis is dead and the factorial collapses to loss × history × activation.

**Falsifier.** Adding the Muon cell to the existing blend (Card 3 protocol) improves fold-paired weighted corr by < 0.0005 after shrinkage → the axis contributes nothing conditional on the panel.

**Cost/serving.** Each kept teacher costs serving time. Only teachers that survive Card 3's conditional-value test get a serving slot or a distillation channel (Card 8). Research-grade full-sequence cells may exceed the endpoint budget; they are authorized only as distillation teachers, flagged `research_only`.

**Descendants / artifacts.** OOF prediction store (per teacher × fold × target, float32, SSD), fold manifest, residual-correlation matrix — reused by Cards 3, 4, 5, 6, 7.

---

## Card 2 — Cross-fitted residual targets: squared-error vs correlation-gradient residuals

**Mechanism.** The second level should be trained on *what the metric actually grades*. Two candidate residual definitions, both cross-fitted:

**(a) Squared-error residual.** With B = clipped-blend prediction from Card 3 (fit on training folds only, applied OOF): r_i = clip(y_i, ±2) − B_i.

**(b) Correlation-gradient residual (proposed).** The pooled weighted correlation for one target, with fixed weights w_i ∝ |clip(y_i, ±2)|, is

ρ̂ = Σ w_i (y_i − ȳ_w)(p_i − p̄_w) / √( Σ w_i (y_i − ȳ_w)² · Σ w_i (p_i − p̄_w)² ).

The gradient of ρ̂ with respect to the prediction p_i, treating y and w as constants, is (proposed derivation):

∂ρ̂/∂p_i = [ w_i (y_i − ȳ_w) − ρ̂ · w_i (p_i − p̄_w) ] / ( σ̂_p √(Σ w_j (y_j − ȳ_w)²) ) / (normalizing choice of scale).

Key structural fact: the gradient is the weighted residual with the *teacher's own weighted projection removed*. So the practical target is: **r_i = ŵ_i (clip(y_i,±2) − B_i) after removing the weighted fit of clip(y,±2) on B within each evaluation pool**, i.e. residualize the naive residual against B itself. A student trained on (a) will partially re-learn B's mistakes in B's own direction; a student trained on (b) is pushed orthogonally to B, which is exactly the direction that moves pooled correlation. **This is proposed inference, to be tested, not a source fact.**

**Algorithm.** Two-stage: (1) fit blend B on train folds, emit OOF B on validation folds; (2) within each validation fold and target, compute r^(a) and r^(b); train student S on current/past features → r with Huber loss. At serve, final = B + λ·S, λ fit on held-out folds only (Card 3).

**Ancestry.** Residual targets inherit blend ancestry; blend teachers' OOF must come from folds disjoint from the student's fitting folds. Two-fold nesting: teachers cross-fit on folds {F1..F5}; blend coefficients fit on {F1,F2,F3} OOF; student and λ fit on {F4,F5} OOF only. Cleanest implementable version: 6 folds, teachers cross-fit on all, blend + λ tuned on folds 1–3 OOF, student fit on folds 4–5 sequences with residual targets built from fold-disjoint teacher OOF.

**Candidate vs control.** Candidate: r^(b) student. Control: r^(a) student, identical architecture/seed/folds. Secondary control: residual target built from a *permuted* blend (breaks conditional signal; any "improvement" there is leakage or selection noise).

**Predicted observable.** r^(b) student raises fold-paired pooled weighted corr more than r^(a), especially on low-|y| rows where clipping saturates and squared-error residuals are dominated by clipping artifacts.

**Cheapest informative test.** Train both students on one fold pair, one seed, small GRU (h=64, L=512). Compare on the untouched fold. ~1 GPU-hour.

**Falsifier.** r^(b) ≈ r^(a) student performance (difference within fold-paired bootstrap CI), or r^(b) student improves local MSE but not pooled weighted corr → the projection removal is not buying conditional signal.

**Cost/serving.** +1 model at serve (or distilled, Card 8). Residual computation at serve needs only B (already produced) — no extra features.

**Descendants.** The r^(b) recipe is the target definition for Card 5's boosting stages.

---

## Card 3 — Static blending: frozen coefficients, selection-overfit controls

**Mechanism.** Blend teacher OOF predictions with weights on the simplex, but treat weight selection itself as an estimator that can overfit folds. Freeze coefficients as artifacts; never let serving code re-derive them.

**Algorithm.** Candidate weight families, fit on blend-tuning folds only (folds 1–3 OOF):
1. **Simplex grid / SLSQP:** maximize pooled weighted corr of Σ α_t P_t, α_t ≥ 0, Σα = 1, per target.
2. **Shrunk simplex:** α = (1−κ)·α̂ + κ·(1/T)𝟙, κ ∈ {0.25, 0.5} — hedge against weight overfit.
3. **Covariance (Markowitz-lens) weights:** from the recorded prior draw (seed 20261008, `research/ideas/claude_20261008_round/passages_selected.json`, Markowitz 1952 p5): *covariance, not individual variance alone, controls weighted-sum variance.* Structural correspondence: teacher OOF **error** covariance Σ̂ (per target, per evaluation pool) plays the role of the asset covariance; weights α ∝ Σ̂⁻¹𝟙 / (𝟙ᵀΣ̂⁻¹𝟙) minimize blend error variance. **Disanalogy, stated:** portfolio returns have a meaningful mean to maximize; here the objective is pooled correlation, not variance, and the "mean" term is absent — so this is minimum-error-variance blending with a correlation objective bolted on. Use only as a candidate family with shrinkage toward equal weights (Ledoit-Wolf-style diagonal shrinkage of Σ̂ **[unverified]**), not as the headline method.

**Information at fit time:** teacher OOF predictions + labels on blend-tuning folds. **At serve time:** none — coefficients are constants in the artifact.

**Candidate vs control.** Each family vs (i) equal weights, (ii) **null blender**: same optimization run on labels with teacher columns independently permuted within fold — the weight fit should find ≈ nothing; realized "gain" on held-out folds is the selection-overfit yardstick. (iii) Best single teacher.

**Predicted observable.** Shrunk simplex ≥ null-blender gain by a margin that survives a fold-paired permutation test; covariance weights ≈ shrunk simplex (they blend similar information) but more stable across fold-pair resamples.

**Cheapest informative test.** Pure CPU, minutes: run all three families on existing teacher OOF (see reuse ledger), evaluate on the untouched fold, plus the permutation control. **No fresh fits required.**

**Falsifier.** Optimized weights beat equal weights on tuning folds but not on the held-out fold by more than the null blender's held-out gain → weight optimization is fitting fold noise; adopt equal weights or a coarse 2–3 teacher blend.

**Cost/serving.** Zero marginal serving cost. This is the highest comparisons-per-hour card in the program.

**Descendants.** Frozen α̂ and Σ̂ are the B in Cards 2, 4, 5, and the distillation target in Card 8.

---

## Card 4 — Learned stacking and causal gates: conditional mixture over teachers

**Mechanism.** Static weights assume one teacher is best everywhere. A gate g(x_t) assigns per-step mixture weights from causal features, exploiting *conditional* diversity: teacher A may dominate high-|y| steps, teacher B the saturated/clipped region.

**Algorithm.** Inputs to the gate (all causal, all cheap): current 112 features (optionally a fixed random projection to 32 dims), step index normalized features (step/20000, and indicators step<99, step in [99, 512]), and teacher disagreement features (pairwise |P_t − P_u| at t, rolling within-sequence mean/std of disagreement computed causally). Gate: softmax MLP, 2×32, trained on blend-tuning folds to predict, per step, the blend weights α*(x) that maximize the weighted-corr surrogate — practically, train with a differentiable soft-mixture loss:

maximize Σ w_i ρ̂( y_i, Σ_t softmax(g(x_i))_t · P_{t,i} )

with temperature regularization so the gate can collapse to static weights when there is no conditional signal. **Gate capacity is deliberately tiny** — a large gate is a leakage/overfit magnet.

**Information at fit time:** labels + teacher OOF + causal features on blend-tuning folds. **At serve:** current features, step, teacher predictions (already computed), causal rolling stats.

**Ancestry.** Gate sees only cross-fitted teacher OOF (never in-sample teacher predictions); gate fitting folds disjoint from the folds used to fit the static blend it must beat; final gate evaluation on the untouched fold.

**Candidate vs control.** Candidate: gate on full input. Controls: (a) gate on features only (no teacher disagreement), (b) gate on permuted inputs, (c) static blend from Card 3. The interesting scientific comparison is (a) vs full: does teacher disagreement carry *conditional* information beyond the features?

**Predicted observable.** Full gate beats static blend on the held-out fold mainly on the |clip(y)| > 1 stratum (unclipped region) and not on the saturated stratum; permuted-input gate gains ≈ 0.

**Cheapest informative test.** CPU-feasible: train the tiny gate on existing OOF with a differentiable correlation surrogate in PyTorch CPU, minutes per fold pair. **No fresh teacher fits.**

**Falsifier.** Gate ≈ static blend after temperature collapse, or gate gains on tuning folds only. Also: gate gain concentrated in one fold → fold-specific, likely noise.

**Cost/serving.** Tiny MLP + causal rolling stats; adds seconds. Rolling stats must be computed with explicit per-sequence state reset.

**Descendants.** Gate weights distribution is a diagnostic for which teachers deserve distillation capacity (Card 8); the disagreement features are reusable in Card 6.

---

## Card 5 — Neural residual boosting with correlation-gradient stages

**Mechanism.** Stage-wise functional gradient boosting over *predictions as functions* (Mason et al., functional gradient boosting **[unverified]**; Friedman's gradient boosting **[unverified]**): each stage fits a small causal model to the negative gradient of the pooled weighted correlation loss evaluated at the current ensemble's predictions, then takes a shrunk step.

**Algorithm.** Initialize F_0 = Card 3 blend (frozen). For stage s = 1..S:
1. Compute pseudo-residuals on boosting-fold sequences: r_i^(s) = −∂ρ̂(y, F_{s−1})/∂p_i evaluated at p = F_{s−1}(x_i) — i.e., Card 2's r

[response interrupted]
