# R3-A: Research Ideas from Optimal Execution, Market Making & Greedy Selection (PDFs)

**Prompt:** Read greedy-method (sparse selection), crypto-MM (inventory-adjusted spreads), and optimal-execution (square-root impact law) papers. Generate 8-12 concrete, falsifiable ideas targeting (a) model diversity/decorrelation, (b) regime shift robustness, (c) symbolic structure of F. Include ≥3 "crazy" ideas. Each idea: 1-line mechanism, document, <1h test, expected outcome.

---

## 1. Temporal Impact Propagation Decomposition

**Mechanism:** Model future price F as sum of exponentially-decaying impacts from recent trades (Gatheral propagator; Doc 3), forcing GRU to learn separate decay rates for bid vs ask. Creates asymmetry baseline—bid-side impact decays faster than ask-side or vice versa.

**Source:** Chicago 2016 Optimal Execution (Gatheral propagator model + square-root law)

**Falsification test (< 1h, 1 GPU):**
- Fit kernel regression to estimate impact decay half-lives separately on bid side (dF) and ask side (events with negative t1).
- Add bid-decay and ask-decay features (exponential basis expansions of recent trades).
- Train baseline GRU vs GRU+decay-features ensemble; compute residual correlation pairwise.
- Expected: Baseline 0.99 corr → ensemble ~0.94–0.96.

**Expected sign:** Reduces residual correlation by forcing bid-ask asymmetry into the model. Modest improvement in score if decay structure is real.

---

## 2. Greedy Ensemble on Market Microclasses

**Mechanism:** Instead of one GRU for all 4 tick-size classes, train separate compact GRUs per class and greedily select best 2–3 for blending. Small universe per class → overfitting teaches orthogonal patterns (Doc 1 greedy selection principle).

**Source:** Greedy Method for SMRPs (grid search, universe size trade-off, overfitting curves)

**Falsification test (< 1h, 1 GPU):**
- Train 4 independent GRUs (one per tick-size class), same hyperparams.
- Compute all 6 pairwise residual correlations across classes.
- Greedily select 2–3 GRUs that minimize max pairwise correlation.
- Ensemble selected GRUs and benchmark against single global GRU.
- Expected: Class-wise GRUs show 0.85–0.90 cross-class residual corr; ensemble ~0.92 global corr.

**Expected sign:** Reduces global ensemble residual correlation below 0.95 because each class learns tick-size-specific microstructure (wide vs tight spreads behave differently). May lose calibration on rare tick sizes.

---

## 3. Inventory-Adjusted Loss Weighting

**Mechanism:** Weight training samples by inferred inventory position (reconstructed from book imbalance + cumulative signed volume). Market makers predict differently when long vs short (Doc 2 inventory feedback in spreads); forces model to learn regime-dependent predictions.

**Source:** Crypto Market Making (inventory-adjusted bid-ask spreads formulas; Doc 2)

**Falsification test (< 1h, 1 GPU):**
- Reconstruct per-snapshot inventory as cumsum(signed-volume) over rolling window, normalized by book depth.
- Reweight MSE loss: L_weighted = MSE * (1 + |inventory|), or per-class reweighting (long vs short).
- Train GRU with weighted vs uniform loss; compare validation curves and residual correlations.
- Expected: Weighted model ~0.97 corr, uniform ~0.99.

**Expected sign:** Inventory conditioning should reduce correlation and improve per-regime calibration. Risk: overfitting on high-inventory regimes if they're rare.

---

## 4. Volatility Regime Masks for Bid-Ask Skew

**Mechanism:** Doc 3 shows market impact scales as sqrt(vol). Split train data into low/med/high realized-vol regimes. Train one GRU with separate prediction heads per regime, or mixture-of-experts gating. Forces model to learn that bid-ask skew *changes* with volatility.

**Source:** Chicago 2016 (square-root law, vol dependence in market impact)

**Falsification test (< 1h, 1 GPU):**
- Segment train by rolling realized vol (20-tick window); define 3 regimes: vol < p33, p33–p66, > p66.
- Train GRU with regime-gated output heads (shared encoder, separate dense heads per regime).
- Measure R² separately per regime and global residual correlation.
- Expected: Baseline 0.99 global corr → gated model ~0.94.

**Expected sign:** Reduces correlation if vol regime is a real confounder. May improve OOS score if test set has different vol distribution than train.

---

## 5. Book-Shape Symbolic Regression Branch (CRAZY)

**Mechanism:** F depends on order-book shape (depth-1, depth-5 imbalances, volume clustering). Append neural-symbolic or tree-based branch that learns *interpretable* micro-rules: "if (depth-1 ask-vol > 2× depth-2) AND (bid-ask imbalance > +X), then F shifts by +Y ticks." Discrete structure adds diversity vs continuous GRU.

**Source:** Chicago 2016 (propagator decay model suggests structure) + Optimal Execution (model-dependent strategies)

**Falsification test (< 1h, 1 GPU):**
- Extract 8–10 book-shape features (imbalances, vol ratios, depth concentration).
- Fit shallow symbolic regression (e.g., SHAKE tree or simple rule extraction) on these features to predict F residuals from GRU.
- Ensemble GRU + symbolic branch.
- Measure ensemble residual corr vs GRU-only.
- Expected: Decorrelation to ~0.93 via discrete structure.

**Expected sign:** Interpretable diversity; may sacrifice accuracy for explainability but should reduce residual corr. Slow to optimize.

---

## 6. Causal Trade-Flow Deconvolution (CRAZY)

**Mechanism:** Instead of recent-trade volume directly, decompose it using impact propagator decay kernel (Doc 3). Split flows into (a) persistent impact (high decay time) and (b) temporary liquidity-seeking (fast mean reversion). Use both as orthogonal inputs to GRU, forcing it to learn different predictive roles for each flow type.

**Source:** Chicago 2016 (Gatheral propagator; impact decay kernel) + Optimal Execution theory

**Falsification test (< 1h, 1 GPU):**
- Estimate impact decay kernel via impulse-response regression on train data (trade → price move over 4–100 ticks).
- Decompose recent-trade volume via Wiener filter into fast and slow components.
- Add both components as separate feature channels to GRU.
- Train jointly; compare single-flow model vs dual-flow model residual corr.
- Expected: Single-flow 0.99, dual-flow ~0.93.

**Expected sign:** Removes confounding between persistent and transient order-flow; should reduce corr substantially if flows are real diversifiers (unlikely but worth testing).

---

## 7. Adversarial Domain-Shift Predictor (CRAZY)

**Mechanism:** Add auxiliary classification head to predict "which market class/instrument am I?" *with gradient reversal*. Forces GRU encoder to learn market-microstructure signals *invariant* to class identity, improving robustness to test-set domain shift. Adversarial training creates diversity via representation learning.

**Source:** Implicit in Doc 3 model-dependence; market-making regimes vary by tick size (Doc 2)

**Falsification test (< 1h, 1 GPU):**
- Add 4-class classification head (tick-size predictor) to GRU encoder.
- Alternate training: 5 steps minimize MSE(t0, t1), then 1 step *maximize* cross-entropy on class prediction.
- Measure OOS performance on held-out class and residual corr.
- Expected: Single-task baseline 0.99 corr; adversarial ~0.92.

**Expected sign:** High risk/reward. Adversarial can either decorrelate strongly or collapse training. If successful, should improve OOS stability on unseen instrument regimes.

---

## 8. Queue Imbalance Discretized States (CRAZY)

**Mechanism:** Queue imbalance is currently continuous. Instead, treat top-20 book levels as discrete "market-maker states" (8 buckets per side = 64 states). Train separate tiny GRU (64 units, 1 layer) on book-state sequences *only*. Blend with main GRU. Market makers think in discrete thresholds; this model captures that logic.

**Source:** Greedy Method (discrete selection principle) + Crypto Market Making (discrete inventory thresholds)

**Falsification test (< 1h, 1 GPU):**
- Discretize bid/ask imbalance into 8 buckets per side; create 64-state encoding.
- Train tiny separate GRU on 64-state sequence (same horizon, different input).
- Simple average ensemble with main GRU.
- Measure ensemble residual corr and per-state prediction accuracy.
- Expected: Tiny GRU learns shallow patterns; ensemble corr ~0.91.

**Expected sign:** Discrete model decorrelates from continuous GRU by design. Works best if market structure is truly quantized at quote levels.

---

## 9. Multi-Scale Temporal Hierarchy

**Mechanism:** Train two GRU heads on *different* event-sampling rates: one every event (baseline), one every 20th event (long-term trend). Blend outputs. Time-scale mismatch forces orthogonal feature extraction: short-term microstructure vs long-term drift (Doc 3 propagator decay across time scales).

**Source:** Chicago 2016 (temporal propagator model; execution over different horizons)

**Falsification test (< 1h, 1 GPU):**
- Build two GRU input streams: (a) every book event, (b) every 20th event sampled into same sequence length.
- Train dual-head GRU: shared encoder, separate heads for each time scale.
- Measure scale-wise residual correlation and global ensemble corr.
- Expected: Single-scale 0.99, dual-scale ~0.93.

**Expected sign:** Should decorrelate via temporal mismatch. Works especially well if regime changes happen at intermediate frequencies (10–100 events).

---

## 10. Market-Impact-Aware Target Adjustment

**Mechanism:** Standard target is realized F. Instead, define adjusted target as (F - impact_penalty), where impact_penalty ~ sqrt(recent_vol / daily_vol) per Doc 3 square-root law. Forces model to predict the "*impact-cleaned*" future price, learning cost compensation rather than raw prediction.

**Source:** Chicago 2016 (square-root law, market impact models, optimal execution cost)

**Falsification test (< 1h, 1 GPU):**
- Compute impact adjustment per snapshot: adjustment = alpha * sqrt(recent_trade_volume / daily_volume).
- Train two GRUs: (a) baseline on F, (b) on (F - adjustment).
- Compare residual correlations and per-regime calibration.
- Expected: Baseline 0.99, impact-adjusted ~0.96.

**Expected sign:** Shifts predictions systematically; decorrelates via target transformation. May improve execution PnL but not validation score; trade accuracy for interpretability.

---

## 11. Learned Position-Aware Modulation Layer

**Mechanism:** Add small neural layer that modulates GRU predictions on-the-fly based on current inferred position (from book imbalance). Model learns to *correct* its prediction if heavily long/short. Learned diversity + feedback loop, not heuristic.

**Source:** Crypto Market Making (inventory feedback formula; Doc 2)

**Falsification test (< 1h, 1 GPU):**
- Extract position signal from book imbalance (5-level cumulative volume ratio).
- Add 32-unit dense layer: modulated_output = GRU_output * (1 + position_factor * position_signal).
- Train end-to-end jointly; compare with baseline GRU.
- Measure residual corr and Sharpe on validation.
- Expected: Baseline 0.99, modulated ~0.98.

**Expected sign:** Conservative; expect small decorrelation (~0.98) but consistent OOS gains if position feedback is real. Interpretable and robust.

---

## 12. Order-Flow Surprise Indicator + Mixture of Experts

**Mechanism:** Define surprise = actual flow - predicted flow (predicted from book shape alone). Predictions should differ under high vs low surprise. Train mixture-of-experts where expert gating depends on flow surprise metric. Captures regime-dependent prediction logic (Doc 2 + Doc 3 flow-dependence).

**Source:** Crypto Market Making (flow is key driver) + Optimal Execution (flow regimes matter)

**Falsification test (< 1h, 1 GPU):**
- Build shallow predictor of trade volume from book features.
- Compute surprise = |actual_volume - predicted_volume| / (predicted_volume + epsilon).
- Train two small GRU experts (low-surprise vs high-surprise regimes).
- Implement soft gating: gate_weight = sigmoid(surprise - threshold).
- Measure per-regime accuracy and global residual corr.
- Expected: Single expert 0.99, MOE ~0.94.

**Expected sign:** Decorrelates via regime-specific experts. Works if flow surprise is a real regime marker; otherwise adds noise.

---

## Summary

**Diversity levers (ranked by expected decorrelation effect):**
1. **Greedy multi-class ensemble** (~0.94 corr): Class-specific overfitting teaches orthogonal patterns.
2. **Discretized queue states** (~0.91 corr): Fundamentally different representation (discrete vs continuous).
3. **Causal flow deconvolution** (~0.93 corr): Orthogonal flow components.
4. **Multi-scale temporal** (~0.93 corr): Different time-scale features.
5. **Adversarial domain shift** (~0.92 corr): Representation learning; high risk.
6. **Volatility regime masking** (~0.94 corr): Regime-dependent heads.
7. **Market-impact-aware targets** (~0.96 corr): Target transformation; modest effect.
8. **Inventory weighting** (~0.97 corr): Loss reweighting; subtle.
9. **Position-aware modulation** (~0.98 corr): Small, learnable feedback loop.
10. **Flow-surprise MOE** (~0.94 corr): Regime gating; depends on surprise quality.
11. **Impact propagation features** (~0.94–0.96 corr): Explicit decay kernel.
12. **Symbolic book-shape rules** (~0.93 corr): Discrete branch; interpretable but slow.

**Next:** Run cheapest tests first (1–2 hours): #2, #4, #9, #8. If any hits >0.92 corr, escalate to ensemble + blending.

**Crazy ideas (#5, #6, #7, #8):** High upside but require careful tuning. Start with #7 (adversarial) if confident in representation learning; #6 (flow deconv) if flow decomposition data available.
