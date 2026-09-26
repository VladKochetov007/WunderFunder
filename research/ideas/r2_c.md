# Round 2 Ideas: Haiku Agent C

Generated 2026-09-26 focusing on model architecture, diversity, distillation, and validation predictiveness under ~80 us/row latency budget (1 core).

---

## 1. Basis Expansion with Residual Links (Shallow Stack)

**Concept + Source:** N-BEATS ICLR 2020 uses deep fully-connected stacks with backward/forward residual links and no time-series-specific components. Showed state-of-the-art on M3/M4 with fast training. Unlike RNNs, FC stacks have low latency when fixed-depth.

**Analogy to Data:** Each row is a 112-dim snapshot; residual links help learn additive basis functions for price and class prediction. Interpretable decomposition (trend + seasonality on order-book features).

**Testable Hypothesis:** A shallow FC stack (3-4 layers, 256 units) with residual paths can match GRU h128 WP while staying <50 us/row and providing different loss landscape for blending.

**Cheapest Test:** `scripts/train_rnn.py --model fc-stack --depth 4 --width 256 --epochs 2` on devB764, compare WP and us/row vs ab20_e2 (0.6707). Single seed sufficient to rule out or advance.

**Falsifier:** If WP <0.65 or latency >55 us/row, the model is too weak for the ensemble.

**Research Tree:** Branch as C4b (alternative to minGRU/LRU); if successful, investigate `--model fc-stack-gru-hybrid` (FC trunk, GRU readout head).

---

## 2. Symbolic-Neural Hybrid with Learned Gating

**Concept + Source:** E-087 symreg found 3 terms (a3, i1_book, i0_book) explain is_scored with corr 0.3375. E-092 nonlinear probes show GBM residual lift only in classes 0/1. Idea: fast symbolic branch + neural residual, soft gating.

**Analogy to Data:** Symreg found interpretable exact rules for scoring mask. Hypothesis: similar exact substructure exists for targets. Fast branch handles obvious cases, neural network focuses on exceptions.

**Testable Hypothesis:** A gated cascade (symreg_pred + α(x) * neural_pred) where α is learned improves blend diversity and reduces inference cost.

**Cheapest Test:** 
1. Fit symreg on ab20_e2 residuals: `scripts/symreg_apply.py --target residuals --max_complexity 5` (2 min).
2. Train learned gate: GBM with 50 trees on [symreg_pred, x] → α, evaluate on devB.
3. Compare blend of (symreg + gate*GRU) vs (baseline + GRU).

**Falsifier:** If symreg branches for t0/t1 have corr <0.08 on residuals or gating adds >2 us/row, abandon.

**Research Tree:** Branch A4a (symreg per class/target with gating); feeds into C3 (distillation).

---

## 3. Cross-Time-Horizon Cascade (Multi-Scale Aggregation)

**Concept + Source:** E-070 shows F reaches its new value within ~1 book event median (lag quantiles 0.1, 0.5, 0.9 = 1, 12, 92 rows). Idea: predict at 1-event, 4-event, 16-event horizons, aggregate.

**Analogy to Data:** Irregular row sampling means one row can span wildly different real times. Multi-horizon auxiliary heads (tree C5: aux-next) treat this as a feature, not a bug.

**Testable Hypothesis:** Auxiliary predictions at next 1, 2, 4 events (with learned aggregation weights) improve signal diversity and help on longer-lag rows where deterministic law breaks down.

**Cheapest Test:** Modify `src/wnn/nn.py` to add 3 extra CE heads predicting `t0/t1 at next_event[k]`. Train with E-084 bootstrap on same devB, measure WP and per-class head AUCs. Budget: 45 min training.

**Falsifier:** If multi-horizon heads overfit (clean-vs-holdout gap >0.002) or reduce main WP, the model is too expressive.

**Research Tree:** Branch C5b (multi-horizon aux heads + learned aggregation); if successful, test on class-3 specifically.

---

## 4. Regime-Adaptive MoE: Quiet vs. Busy Experts

**Concept + Source:** Tree B4 notes WP 0.36 on rows >=170 quiet (stale regime), 1/3 of weight. Tree C2 plans regime MoE. Hypothesis: separate lightweight expert for quiet regime (cheaper compute, less capacity).

**Analogy to Data:** Quiet regime has no new price changes, only constant features. Busy regime has frequent updates. A soft router learns p(quiet | state) and routes to expert A (simple) or B (richer).

**Testable Hypothesis:** A soft MoE (budget: router 5 us, expert_quiet 20 us, expert_busy 40 us) reduces latency and improves class-3 WP by capturing regime-specific regularities.

**Cheapest Test:** 
1. Train binary regime classifier on heur_valid: quiet = (rows_since_last_price_change > 100).
2. Fit expert_A (2-layer GRU h64) on quiet rows only, expert_B (h128) on normal.
3. Train soft router (1 FC layer) on full data.
4. Measure WP on devB by regime. Budget: 60 min.

**Falsifier:** If router latency + overhead exceeds savings, or if per-regime WP is flat, the MoE is not worth it.

**Research Tree:** Branch C2a (regime MoE with quiet/busy); feeds into C3 (blend efficiency).

---

## 5. Learned State-Reset Controller (Meta-Learning Event Clock)

**Concept + Source:** E-081 shows depth=1 and depth=3 are within noise (~0.002), suggesting the GRU state carries little long-range dependency. Idea: learn WHERE to reset state (event-driven, not time-driven) and WHERE to accumulate.

**Analogy to Data:** Sequences contain multiple sub-regimes (quiet, busy, regime change after a huge move). A meta-learner observes activity (a2, a3, a5, price-change indicator) and decides: reset state now or keep?

**Testable Hypothesis:** A learned reset gate r(x) ∈ [0, 1] applied at each step (new_state = (1 - r) * old_state + r * new_candidate_state) improves WP on class-3 by learning when to "forget the past."

**Cheapest Test:** Modify GRU forward pass to add reset gate: `r = sigmoid(fc_layer(x))`. Train with frozen GRU weights first (measure r patterns), then fine-tune. Measure WP vs ab20_e2. Budget: 30 min.

**Falsifier:** If reset gate learns r ≈ 0 (never reset) or r ≈ 1 (always reset), the mechanism is not being used.

**Research Tree:** Branch C1a (learned state reset); informs tree A5 (time order recovery) if reset patterns reveal hidden sequencing.

---

## 6. Lookahead Label Calibration (Validation Shift Robustness)

**Concept + Source:** E-077 shows validation overstates test by ~0.022. E-084 metric anatomy shows between-seq variance is 18-20% of weighted variance. Idea: use known validation-set label distribution to calibrate predictions on later test data with unknown (softer) mask.

**Analogy to Data:** Test mask is softer (E-084: critic lambda=0.02 vs oracle). Validation labels under one mask distribution don't generalize to a softer mask. Use the label histogram from validation to reweight predictions.

**Testable Hypothesis:** A calibration function that maps validation-per-seq statistics (mean, variance of predictions) to test-predicted statistics (via moment matching on known valid data) reduces validation-test gap.

**Cheapest Test:** 
1. Compute per-seq mean/variance of (baseline + GRU blend) on devA, devB, holdout.
2. Fit isotonic regression: per-seq_mean_valid → expected_per-seq_mean_test (on holdout).
3. Re-evaluate blend on holdout using adjusted per-seq means. Budget: 20 min.

**Falsifier:** If calibrated holdout is within 0.001 of uncalibrated (no shift), the mask difference is not the driver.

**Research Tree:** Branch D3a (recency + mask-aware weighting); addresses goal robustness.

---

## 7. Anchor-Time Feature Re-Engineering (Price-Change Clock)

**Concept + Source:** E-091 decoded a5 (trade clock) and a6 (book clock). E-079 found F changes at book events. Idea: engineer new anchor time = rows since LAST TARGET CHANGE, recalibrate all lagged/rate features relative to this.

**Analogy to Data:** Features like "rate_book_hl64" measure activity over a fixed window, but target changes are irregular. If we anchor to price-change events, lagged activity becomes more predictive.

**Testable Hypothesis:** A new feature "rows_since_target_change" + interaction terms with rate/depth features recover correlation signal that fixed-window features miss, yielding +0.001-0.002 linear lift on GRU residuals.

**Cheapest Test:** Engineer feature in heuristic library, run `scripts/lift_probe.py --groups anchor_time` on ab20_e2. Budget: 15 min.

**Falsifier:** If linear lift is still zero (per E-088 finding on 114 heuristics), price-change clock is not a missing feature.

**Research Tree:** Branch A2a (time-anchor feature engineering); feeds A4 (symreg per class).

---

## 8. Cross-Sequence Embedding & Shallow Meta-Learning

**Concept + Source:** E-084 shows per-seq level variance is 18-20% of total. Hypothesis: some sequences are structurally easier (high class 0/1 fraction, frequent moves) or harder. Idea: learn sequence embedding from first 100 rows, modulate GRU capacity.

**Analogy to Data:** Each sequence is a "new instrument" (tree F4) with hidden identity. Embedding captures sequence difficulty, class distribution, regime.

**Testable Hypothesis:** Sequence embeddings (learned from row 0-100 only, frozen after) + shallow meta-network (1-2 FC layers) that output GRU bias modulations improve oracle-vs-learned correlation per sequence.

**Cheapest Test:**
1. Encode first 100 rows of each train sequence into 16-dim embedding via separate shallow GRU.
2. Freeze embeddings, train main GRU with modulated biases: GRU_biases += fc_meta(seq_embedding).
3. Measure per-seq oracle correlation improvement vs plain GRU. Budget: 50 min.

**Falsifier:** If oracle correlation improves <1% or embeddings collapse (near-zero variance), the mechanism is not engaged.

**Research Tree:** Branch C2b (per-sequence meta-learning); informs A5 (instrument identity).

---

## 9. Distilled Blend via Hypernetwork (Inference Cost Reduction)

**Concept + Source:** E-085 shows blend (baseline + GRU) achieves 0.6869 holdout but costs 90+ us/row. Goal: compress to single model at GRU latency (~42 us/row).

**Analogy to Data:** Blend output is a learned mixture; hypernetwork idea: train shallow FC to map sequence features → modulation parameters that make a single GRU match blend output.

**Testable Hypothesis:** Knowledge distillation to a single h192 GRU (with hypernetwork-modulated parameters per-sequence) can match 0.685+ while staying <50 us/row.

**Cheapest Test:**
1. Generate blend outputs on all train sequences: `cache/blend_preds_train.npz`.
2. Train 1-layer hypernetwork: seq_stats → Δ_weight matrices for GRU.
3. Distill GRU to match blend via MSE on outputs. Budget: 80 min training.

**Falsifier:** If distilled WP <0.682 or final latency >55 us/row, hypernetwork is insufficient capacity.

**Research Tree:** Branch C3a (knowledge distillation with hypernetwork modulation); enables final production model.

---

## 10. Class-3 Event Model + Analytic Readout Layer

**Concept + Source:** E-082 shows next-move direction AUC = 0.99 within 2 rows, 0.62 far from moves. E-079 shows exact class-3 labels = round(mean of 4 book events). Idea: lightweight event classifier (HGB) predicts next move direction, feeds into analytic 3-way readout layer.

**Analogy to Data:** Class 3 follows a deterministic law once you know which side (bid/ask/mid) the price goes. A fast event classifier replaces GRU's soft prediction, exact formula handles magnitude.

**Testable Hypothesis:** A hybrid model (GRU for classes 0-2, HGB-based for class 3) with tied features achieves 0.45+ on class 3 WP while staying <55 us/row.

**Cheapest Test:**
1. Train HGB (150 trees, d=3) on class-3 rows: x → next-move direction.
2. Build readout layer: [GRU_hidden, class_logits, HGB_pred] → [t0, t1] via linear layer.
3. Measure WP on class 3 vs ab20_e2 (0.4463). Budget: 45 min.

**Falsifier:** If class-3 WP <0.45 or total latency >60 us/row, the hybrid is not worth it.

**Research Tree:** Branch B2a (class-3 event model + exact readout); targets class-3 bottleneck directly.

---

## 11. Stale-Row Regime Detector with Conditional Prediction

**Concept + Source:** Tree B4: WP 0.36 after >170 quiet rows (1/3 of weight). Idea: separate lightweight model for stale regime; conditional prediction: if stale, use stale-model else use main GRU.

**Analogy to Data:** Stale rows have no target changes, only constant features. Simple linear model may work better; saves GRU capacity for normal rows.

**Testable Hypothesis:** A gated model (regime indicator + expert_stale vs expert_normal) improves WP by 0.002-0.003 and reduces latency by 5-10% on average.

**Cheapest Test:**
1. Define stale: rows_since_last_price_change > 100.
2. Train expert_stale: 1-layer GRU h32 on stale rows only; expert_normal: ab20_e2.
3. Soft gate: logistic(fc_gate(x)) routes. Measure WP on devB, breakdown by regime. Budget: 60 min.

**Falsifier:** If stale-regime WP does not improve or gate overhead exceeds savings, abandon.

**Research Tree:** Branch C2c (stale-regime gating); validates tree B4 finding.

---

## 12. Dual-Path Information Bottleneck (Diversify via Decorrelation)

**Concept + Source:** E-084 shows blend (baseline + GRU) adds +0.0017 on devA/devB. Hypothesis: baseline carries mode/level info, GRU should focus on residuals. Idea: explicitly train two decorrelated paths.

**Analogy to Data:** Baseline is slow, smooth, captures global level. GRU should predict orthogonal residuals. Information bottleneck loss (L_ib) encourages paths to be independent.

**Testable Hypothesis:** A dual-head GRU (path A supervised on baseline residuals, path B on clean targets) with correlation penalty learns non-redundant signals, improving blend from +0.0017 to +0.003.

**Cheapest Test:**
1. Modify nn.py to add path_A head (supervised on baseline residuals).
2. Add loss term: L_ib = -corr(path_A_pred, path_B_pred) (encourage anti-correlation).
3. Train with α_ib ∈ [0.1, 0.5], measure WP on devB and correlation of two heads. Budget: 45 min.

**Falsifier:** If correlation stays >0.8 (paths remain similar) or WP decreases, dual-path is not effective.

**Research Tree:** Branch C2d (decorrelated dual-path ensemble); informs C3 (distillation strategy).

---

## Summary

All 12 ideas address gaps in the current ensemble:
- **Diversity:** Ideas 1, 2, 4, 8, 10, 12 propose genuinely different model structures.
- **Latency:** Ideas 1, 4, 9 explicitly target <55 us/row inference.
- **Validation drift:** Idea 6 directly tackles test-vs-valid shift.
- **Class-3 bottleneck:** Ideas 4, 10, 11 focus on the low-WP regime.
- **Distillation:** Idea 9 compresses blend into single model.
- **Interpretability:** Ideas 2, 7, 10 leverage symbolic/causal structure.

Recommended test order: 1 → 4 → 10 (architecture diversity) → 6 (validation) → 9 (compression) → 2 (hybrid).
