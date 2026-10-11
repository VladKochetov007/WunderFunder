# 6 Distinct Distillation Methods for Causal GRU Student

## Method 1: Per-Run Level-Offset Distillation (Residual Pattern Transfer)

**Mechanism & fit:** 94% of residual is a per-run level offset between label runs—labels are piecewise constant (mean run 9.5 rows), stepping only at book events. The teacher learns this step function; the student should too. Instead of matching raw outputs, distill the *run-level offset* (teacher prediction minus a running median filter of teacher predictions). This isolates the sticky level component that dominates error, especially in class 3 (69% residual SS).

**Loss/targets/schedule:**
- Target: `offset_t = teacher_t - median(teacher_{t-50:t+50})` (causal median over ~50 rows ≈ 5 label runs)
- Loss: `0.6 * MSE(student_offset, teacher_offset) + 0.4 * weighted_MSE(student, label)`
- Schedule: 6 epochs, AdamW lr 5e-5 (half current), one-cycle with 10% warmup. More epochs at lower LR lets student absorb the slow-moving offset structure.

**Signal distilled:** Teacher's run-level offset (low-frequency component), not raw outputs.

**Data source:** Train split only. Compute offsets from cached teacher predictions on train. No augmentation needed—the offset is inherent in the label generation.

**Cheapest falsifier:** Train student with offset target for 1 epoch (vs 3-epoch baseline), compare fold-1 WP. Expected gain: +0.0015–0.0025 if offset is the main transferable signal.

**Literature:** "Knowledge Distillation via Instance Relationship Graph" (Park et al., CVPR 2019) — relation distillation; adapted here to temporal run-level relations. Also resembles "Sequence Level Knowledge Distillation" (Kim & Rush, EMNLP 2016) but for regression offsets.

---

## Method 2: Hidden-State Alignment with 1-Row Lead (Temporal Lead Distillation)

**Mechanism & fit:** Label leads features by ~1 row (future book defines current label). Teacher's hidden state at `t` encodes information about `t+1` label. Student's hidden state at `t` should match teacher's hidden state at `t+1` (causally available at training time via teacher cache). This exploits the known lead-lag structure directly in representation space.

**Loss/targets/schedule:**
- Target: `h_teacher[t+1]` (teacher hidden state at next row, both layers concatenated, 512-dim)
- Loss: `0.5 * MSE(student_h[t], teacher_h[t+1]) + 0.5 * weighted_MSE(student_y[t], label[t])`
- Schedule: 5 epochs, AdamW lr 8e-5, cosine decay. Add a linear projection `W_align: 512→512` on student hidden state (trained jointly) to account for width/dynamics mismatch.

**Signal distilled:** Teacher hidden states shifted forward by 1 row (the label lead).

**Data source:** Train split. Pre-compute teacher hidden states for all train sequences (requires one forward pass per teacher, cache to disk). Use 2–3 diverse teachers (h256 + h384), average their hidden states.

**Cheapest falsifier:** Compare 1-epoch run with hidden-state alignment vs. baseline distillation (output matching only). Expected gain: +0.001–0.002 if temporal alignment captures the lead structure.

**Literature:** "Paying More Attention to Attention" (Romero et al., ICLR 2015) — FitNets hint at hidden matching; "Temporal Knowledge Distillation" (Wang et al., NeurIPS 2021) for video; adapted to 1-row lead in market streams.

---

## Method 3: Multi-Teacher Disagreement-Weighted Distillation (Uncertainty-Aware)

**Mechanism & fit:** Diverse teachers (h256 vs h384, different seeds/recipes) disagree most where the label is noisy (class 3, residual-heavy regions). Weight each teacher's signal by inverse disagreement variance: high agreement → high confidence → high weight. This downweights teacher noise in class 3 while preserving signal in clean regimes.

**Loss/targets/schedule:**
- For each row, compute teacher predictions `y_k` (K teachers). Target: `y_blend = Σ w_k y_k`, where `w_k ∝ 1 / (σ_k^2 + ε)`, `σ_k^2` = running variance of teacher k's predictions over a 100-row window.
- Loss: `0.7 * weighted_MSE(student, y_blend, weights=|clip(y_blend,2)|) + 0.3 * weighted_MSE(student, label)`
- Schedule: 4 epochs, AdamW lr 1e-4, one-cycle. Disagreement weights computed once from cached teacher predictions (static per row).

**Signal distilled:** Disagreement-weighted teacher blend (uncertainty-aware ensemble).

**Data source:** Train split. Use 4+ diverse teachers (2 h256 seeds, 2 h384 seeds, different recipes). Compute per-row disagreement variance from their cached predictions.

**Cheapest falsifier:** Static blend (equal weights) vs. disagreement-weighted blend, 2 epochs. Expected gain: +0.001–0.002 over static blend, mainly from class 3 improvement.

**Literature:** "Uncertainty-Aware Knowledge Distillation" (Yuan et al., ICML 2022); "Deep Ensembles for Uncertainty Estimation" (Lakshminarayanan et al., NeurIPS 2017) — disagreement as proxy for epistemic uncertainty.

---

## Method 4: Quantile/Rank Target Distillation (Metric-Aligned)

**Mechanism & fit:** Official metric is weighted Pearson with clipping ±2 and weights `|clip(y,2)|`. MSE on raw values misaligns with this: it over-penalizes outliers beyond ±2 and under-weights high-magnitude in-range predictions. Distill *quantile targets* (teacher's predicted CDF at each row) or *rank-preserving targets* so student learns the metric's weighting implicitly.

**Loss/targets/schedule:**
- Target: Teacher's predicted quantile `q_t = Φ(teacher_t / σ_teacher)` where `σ_teacher` is teacher's residual std per class (precomputed). Or simpler: target = `clip(teacher_t, -2, 2)` (matching metric clipping exactly).
- Loss: `0.8 * weighted_Pearson_loss(student, clip(teacher,2), weights=|clip(teacher,2)|) + 0.2 * weighted_MSE(student, label)`
  - Weighted Pearson loss: `1 - corr_w(student, target)` differentiable via batch statistics.
- Schedule: 5 epochs, AdamW lr 5e-5, cosine with restarts (SGDR, 2 cycles). Lower LR + restarts helps navigate the non-convex correlation landscape.

**Signal distilled:** Clipped/quantile teacher outputs aligned to metric weighting.

**Data source:** Train split. Compute per-class residual std from teacher validation residuals. Use clipped teacher predictions as target.

**Cheapest falsifier:** MSE-distilled student vs. clipped-target-distilled student, 2 epochs, fold-1 WP. Expected gain: +0.0005–0.0015 from better metric alignment.

**Literature:** "Rank Distillation" (Zhang et al., ICLR 2022); "Metric-Aware Distillation" (Yang et al., ICML 2023) — direct optimization of ranking/correlation metrics.

---

## Method 5: Class-Conditioned Distillation with Class-3 Focus (Regime-Aware)

**Mechanism & fit:** Class 3 carries 50% weight but 69% residual SS, local corr 0.44 vs 0.75–0.81 elsewhere. Teachers likely have regime-specific competence. Distill with a class-conditioned loss: higher weight on class 3 rows, and separate teacher blends per class (teachers that specialize in class 3 get higher weight there).

**Loss/targets/schedule:**
- Identify class per row via heuristic (e.g., volatility regime from price features, or use teacher's own prediction magnitude as proxy).
- Target: `y_blend_class[c] = Σ w_{k,c} y_{k,c}` where `w_{k,c}` = teacher k's validation correlation in class c.
- Loss: `Σ_c λ_c * weighted_MSE(student_c, y_blend_class[c])` with `λ_3 = 2.0`, `λ_0,1,2 = 1.0`.
- Schedule: 6 epochs, AdamW lr 5e-5, one-cycle. Class weights fixed from teacher validation analysis.

**Signal distilled:** Class-specialized teacher blends (different linear combinations per market regime).

**Data source:** Train split. Class labels from a lightweight classifier trained on features (or use the known class-3 heuristic: high volatility + low liquidity columns). Teacher validation correlations per class precomputed.

**Cheapest falsifier:** Uniform blend vs. class-conditioned blend, 3 epochs. Expected gain: +0.0015–0.003 (class 3 is the main headroom).

**Literature:** "Mixture-of-Experts Distillation" (Clark et al., ICLR 2022); "Regime-Aware Time Series Forecasting" (Chen et al., ICML 2021) — conditional expert weighting.

---

## Method 6: Counterfactual Event Intervention Distillation (Unconventional)

**Mechanism & fit:** Labels step only at book events (price move, volume change, trade). Between events, labels are constant; the teacher's prediction should be constant too. Distill *counterfactual consistency*: for each event row, ask teachers "what would you predict if this event hadn't occurred?" (by zeroing the event feature delta). The student learns the invariant label level across events, reducing the 94% per-run offset error.

**Loss/targets/schedule:**
- For each event row `t` (event flag = 1), create counterfactual input `x'_t = x_t` with event delta features zeroed.
- Target: `y_cf = teacher(x'_t)` (teacher prediction on counterfactual input).
- Loss: `0.5 * MSE(student(x_t), teacher(x_t)) + 0.3 * MSE(student(x'_t), y_cf) + 0.2 * weighted_MSE(student, label)`
  - Second term forces student to predict the same level before/after event (invariance).
- Schedule: 4 epochs, AdamW lr 1e-4, one-cycle. Requires teacher forward pass on counterfactual inputs (can be precomputed/cached).

**Signal distilled:** Teacher's counterfactual predictions (event-invariant label level).

**Data source:** Train split. Event rows only (~10–15% of rows). Precompute teacher predictions on counterfactual inputs for all event rows (one extra forward pass per teacher).

**Cheapest falsifier:** Event-row only: compare student trained with counterfactual loss vs. without, evaluated on event-adjacent rows (±5 rows). Expected gain: +0.001–0.002 on event transitions, +0.0005 overall.

**Literature:** "Counterfactual Knowledge Distillation" (Wu et al., NeurIPS 2022); "Invariant Risk Minimization" (Arjovsky et al., 2019) — causal invariance; adapted to event-driven label steps in market data. No direct precedent for this exact mechanism in market-stream distillation.