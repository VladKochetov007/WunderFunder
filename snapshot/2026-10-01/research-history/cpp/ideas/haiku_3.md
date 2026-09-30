# Haiku-3 Idea Generation: Trading Concepts → Order-Book Prediction

**Source documents:** "TWAP & VWAP - More important than you think" (VERTOX); "A simple but powerful measure of market efficiency" (Tran & Leirvik, 2019)

**Transfer mechanism:** TWAP/VWAP prevent manipulation via temporal averaging; AMIM measures autocorrelation-driven inefficiency. Both suggest: (a) detectability of order flow patterns, (b) time-varying efficiency regimes, (c) volume-weighted aggregation beats raw price/count.

---

## Idea 1: Volume-Weighted Micro-Price as Primary Feature

**Concept (VWAP, p. 3-4):** VWAP weights prices by volume; harder to manipulate than TWAP because size matters. Settlement on VWAP prevents end-game price spikes.

**Analogy:** Current micro-price ≈ (bid × vol_ask + ask × vol_bid) / (vol_bid + vol_ask). Predict future volume-weighted micro-price instead of simple mid-price.

**Hypothesis:** VW micro-price is a smoother, more manipulable-resistant signal than unweighted mid. Our GRU learns faster on this target; WP gains +0.005-0.010.

**Cheapest test:** 
- Extract level volumes (L1-L5) from features; compute VW micro-price for each row.
- Retrain GRU with targets = future VW micro at t+k vs current targets.
- Compare validation WP (t0, t1) against baseline.

**Falsification:** VW micro-price has lower variance (smoother) but GRU WP **decreases** because information loss outweighs noise reduction.

**Branch:** B1 (heuristic library) or A2 (feature engineering).

---

## Idea 2: Autocorrelation-Driven Loss Weighting

**Concept (AMIM, p. 70-71, 92-94):** AMIM uses absolute autocorrelation |β| to measure inefficiency. High ACF = market predictable but hard (anticorrelated moves). Low ACF = random/efficient.

**Analogy:** Compute rolling autocorrelation of GRU residuals (prediction - target). High ACF regions = model misses patterns; downweight their loss or upweight their learning.

**Hypothesis:** Rows where residuals are autocorrelated indicate **local model blindness**. Upweighting their gradient loss forces the GRU to capture missed structures. WP +0.008-0.012.

**Cheapest test:**
- Train baseline GRU; extract residuals on valid.
- Per-sequence, compute rolling ACF(lag=1) of residuals.
- Retrain with loss = MSE * (1 + ACF magnitude), clipped.
- Compare valid/test WP.

**Falsification:** ACF weighting is **statistically noise** (random walk, no lag structure). Weighted loss = same as unweighted; or WP **decreases** due to overfitting to noise.

**Branch:** D2 (weights: label uniqueness, propensity) or C1 (GRU recipe).

---

## Idea 3: Efficiency Regime Classifier (AMH-style)

**Concept (AMIM, p. 56-59, 410-424):** Adaptive Market Hypothesis: markets are inefficient (predictable) in crisis, efficient in calm. AMIM varies over time; crashes spike AMIM.

**Analogy:** Build a rolling classifier: is this sequence's orderbook in a "high-efficiency" (random) or "low-efficiency" (predictable) regime? Upweight GRU on low-efficiency rows; separate heads or Bayesian epistemic weights.

**Hypothesis:** Class-3 is a pseudo-crisis regime (large ticks, jumpy). Detecting local inefficiency (via ACF or variance ratio) lets us dial up prediction confidence. Sequence-level stratification improves WP by +0.006-0.015.

**Cheapest test:**
- Compute per-sequence: rolling variance, autocorrelation, Hurst exponent (low=random, high=trending).
- Cluster sequences into 2-3 efficiency regimes (K-means).
- Retrain GRU with mask: upweight loss for "inefficient" sequences.
- Check per-regime WP; does WP improve where regimes are well-separated?

**Falsification:** Efficiency regimes are **sequence-static** (not predictive of local row difficulty). Masking by regime = same as no masking; or test WP **worse** due to label leakage.

**Branch:** A3 (scoring mask) or B4 (stale-row regime).

---

## Idea 4: Order-Flow Momentum with Exponential VWAP Decay

**Concept (VWAP, p. 4, recursive formula; TWAP execution, p. 5-6):** VWAP is recursive. Execution algorithms vary size proportionally to distance from running VWAP. Implies: recent volume matters more than old.

**Analogy:** Instead of fixed 20-row lookback, use exponentially decayed VWAP of **net order flow** (cumsum of buy-sell imbalance weighted by volume). Replace static features with this decaying momentum signal.

**Hypothesis:** Exponential decay (half-life ~5-10 rows) captures microstructure memory better than step lookback. Decaying momentum outperforms fixed-window features. +0.003-0.008 WP gain.

**Cheapest test:**
- Extract net volume flow from trade columns (dp, dv).
- Compute running decayed sum: momentum[t] = λ * momentum[t-1] + (1-λ) * flow[t], λ ~ 0.7.
- Add momentum feature to GRU input.
- Compare against baseline (no feature).

**Falsification:** Decay constant λ is **arbitrary**; different λ yields no consistent improvement. Or momentum feature is **redundant** with raw trade volume already in model.

**Branch:** B1 (heuristic library) or B3 (i1 cross-instrument).

---

## Idea 5: VWAP-Style Settlement Target (k-Event Averaging)

**Concept (VWAP settlement, TWAP execution, p. 3-6):** Derivatives don't settle on single price; they settle on VWAP/TWAP average. This smooths noise, prevents manipulation.

**Analogy:** Instead of predicting next-row price move (t0, t1), predict the **average** move over the next k book events (k ~ 2-4). Smooth target = lower noise, stronger signal.

**Hypothesis:** Targets with k-event VWAP smoothing have lower label noise; GRU overfits less; validation WP is closer to test WP (lower generalization gap). WP +0.004-0.010.

**Cheapest test:**
- Create new targets: t0_smooth = mean(t0[t:t+k]), t1_smooth = mean(t1[t:t+k]) for k=2,3,4.
- Retrain GRU on smoothed targets.
- Compare valid WP and valid/test divergence.
- Check per-class effect (class-3 bottleneck might shrink).

**Falsification:** Smoothing **removes discriminative information**. Test WP **decreases** more than train WP improves. Or smooth targets are **perfectly correlated** with original (no noise reduction).

**Branch:** A1 (target law) or D2 (weights, label quality).

---

## Idea 6: Detectability Score & Randomization of Feature Noise

**Concept (Execution detection, TWAP vs VWAP, p. 5-6):** Constant TWAP execution is easy to frontrun; VWAP is harder to detect. Random jitter in order placement evades prediction.

**Analogy:** Compute a "detectability score" per sequence: how regular/predictable is the inter-event time distribution? Score → use as auxiliary target (predict regularity). If sequences are **detectable** (regular), model should learn that target moves are **harder to predict** (because they're being front-run).

**Hypothesis:** Auxiliary head predicts row-level regularity (AUC ~0.6-0.7). Joint training on {t0, t1, regularity} forces model to separate "frontrun noise" from true signal. WP +0.005-0.012 for detectable seqs.

**Cheapest test:**
- Per sequence, compute inter-event time variance (coefficient of variation).
- High CV = random (hard to frontrun) → label regularity = 0.
- Low CV = deterministic (easy to frontrun) → label regularity = 1.
- Add auxiliary CE/MSE head for regularity.
- Check: does GRU learn to predict regularity? Does auxiliary loss help main WP?

**Falsification:** Regularity is **uncorrelated with label noise** or **independent of frontrun risk**. Auxiliary head learns nothing; main WP **unchanged or worse**.

**Branch:** B2 (class-3 event classifier) or F2 (privilege-teacher denoising).

---

## Idea 7: Market Inefficiency Index (MII) as Confidence Modulation [CRAZY]

**Concept (AMIM, p. 99-117):** AMIM measures standardized autocorrelation magnitude via CI adjustment. AMIM > 0 = inefficient (predictable), AMIM < 0 = efficient (random).

**Analogy:** Compute per-sequence AMIM-style metric using **GRU prediction residuals** (not price returns): MII[seq] = |autocorr of residuals| / CI. Use MII to **gate model confidence**: output = base_pred * (1 + α * MII), α ~ 0.05. Inefficient seqs → higher confidence; efficient seqs → lower confidence, closer to baseline.

**Hypothesis:** Confidence modulation reduces overfitting in efficient (hard) sequences. Model learns to "back off" when residual autocorr is low (marker of missing patterns). WP +0.008-0.015, especially in class 3 (+0.02-0.03).

**Cheapest test:**
- Train baseline GRU; extract residuals on valid.
- Per sequence, compute AMIM-style inefficiency index.
- Modulate GRU outputs: pred_final = pred_base * (1 + 0.1 * (MII - MII_mean)/MII_std).
- Compare valid WP (per-class breakdown).

**Falsification:** MII is **sequence-correlated with class label** (not a new signal). Confidence modulation = implicit class weighting already done. Or MII **noise**: test WP **decreases** due to random multiplier.

**Branch:** F3 (privileged-teacher label denoising) or B4 (stale-row regime).

---

## Idea 8: Anti-Detectability Regularization [CRAZY]

**Concept (Execution detection, p. 5-6):** Regular order placement is frontrun-able. VWAP adds randomness to avoid detection. Implication: if order placement is random, predictability is lower (less front-run interference).

**Analogy:** Add a **regularizer** that penalizes the GRU if its predictions become too "regular" (i.e., high autocorrelation in predictions). Regularizer = λ * E[|ACF(pred, lag=1)|]. This forces the model to output noisy, harder-to-exploit predictions, avoiding the fate of a detectable algo.

**Hypothesis:** Prediction regularizer reduces adversarial predictability (a hypothetical attacker trying to predict **our predictions**). Boosts robustness; WP +0.010-0.020, especially on test vs validation divergence (overfitting reduction).

**Cheapest test:**
- Compute ACF(lag=1) of GRU outputs on valid in mini-batches.
- Add reg term: loss = main_loss + 0.01 * mean(|ACF|).
- Retrain GRU.
- Compare valid WP, train/valid loss curves (should see less overfitting).

**Falsification:** Regularizer is **orthogonal to order-book dynamics**. WP **decreases** because model is forced to ignore true patterns. Or ACF of **noise already zero** (GRU predictions already random).

**Branch:** F5 (predict baseline error / stack on residual) or E1 (fused blend inference).

---

## Idea 9: Multi-Horizon VWAP Targets (Pyramid)

**Concept (VWAP computation, p. 3-4, and execution horizons, p. 5-6):** VWAP varies by time horizon (5 min vs 12 hour). Different horizons give different prices, all valid.

**Analogy:** Instead of t0, t1 = single-horizon targets, predict a **pyramid**: t0_1step, t0_2step, t0_4step, t1_1step, ... (powers of 2 or Fibonacci steps ahead). Supervise all 6-8 heads jointly. Predictions at different scales → richer signal; early layers learn fine structure, later layers coarse trends.

**Hypothesis:** Multi-horizon supervision forces GRU to learn hierarchical time-scales. Test WP +0.012-0.025 because model learns to blend short/medium-term forecasts (akin to ensemble of horizons).

**Cheapest test:**
- Precompute targets at t+1, t+2, t+4, t+8 steps.
- Clone GRU heads: 6 output heads for (t0_1, t0_2, t0_4, t1_1, t1_2, t1_4).
- Weight all heads equally in loss (or learn weights via auxiliary gating).
- Inference: blend outputs (e.g., avg or learned weights).
- Compare test WP against baseline (single-head).

**Falsification:** Multi-horizon targets are **collinear** (high correlation, no new info). Multi-head model = same as single-head. Or test WP **unchanged or worse** due to diluted gradient signal.

**Branch:** C4 (minGRU/LRU/attention over last-k) or A1 (target law).

---

## Idea 10: Per-Class Efficiency Adaptation (Class-3 Asymmetry)

**Concept (AMIM crisis effects, p. 410-424; Tran & Leirvik finding):** Large-cap efficient (AMIM low), small-cap inefficient (AMIM high). Efficiency varies **dramatically** by asset.

**Analogy:** Our data: Class 3 is inefficient (WP 0.42-0.44 vs 0.72-0.79 for others). Build **class-conditional GRU**: separate parameter sets for each class, or a shared GRU with class-gated adapter layers. Class 3 gets richer hidden state (h256) and 3 epochs; classes 0-2 get h128, 2 epochs. Hypothesis: efficiency-matched capacity → better generalization.

**Hypothesis:** Asymmetric model capacity per class improves class-3 WP by +0.025-0.050 without hurting others. Class 3 has more expressivity to capture "hard" inefficient patterns.

**Cheapest test:**
- Train 4 separate GRU models (one per class) with hyperparameters tuned per class.
- Inference: route each test sequence to its class-specific model.
- Compare test WP (per-class and overall) vs. shared model.
- Measure per-class WP; check if class-3 gap shrinks.

**Falsification:** Class 3 is hard due to **label noise or data scarcity**, not inefficiency. Separate models overfit per class. Test WP **unchanged or worse** due to reduced training set per model.

**Branch:** C1 (GRU recipe) or A5 (shift across train/valid/test).

---

## Idea 11: Volatility Regime as Co-Predictor (Variance Ratio)

**Concept (Tran & Leirvik, p. 440):** AMIM compared to "variance ratio" (another efficiency measure). Variance ratio = Var(k-step returns) / k * Var(1-step returns). Ratio <1 = mean-reverting, >1 = trending.

**Analogy:** Compute per-sequence variance ratio (VR) for order-book mid-price moves. High VR = trending (autocorr > 0), Low VR = mean-reverting (autocorr < 0). Use VR as an auxiliary input or gating signal: GRU learns differently depending on regime. Joint head: predict (t0, t1, VR).

**Hypothesis:** VR co-prediction helps model distinguish trending vs. mean-reverting regimes. WP +0.010-0.020, especially for classes 0-2 (more variance ratio variation than class 3).

**Cheapest test:**
- Compute historical VR(lag=2 or 4) on rolling windows of mid-price.
- Add VR as input feature + auxiliary target (CE on VR regime).
- Retrain GRU.
- Compare validation WP and per-class breakdown.

**Falsification:** VR is **already captured** by order-book features (raw price momentum). Auxiliary head learns nothing; WP **unchanged**. Or VR **random walk metric** (uninformative for single-sequence prediction).

**Branch:** B1 (heuristic library) or A4 (symbolic explanations of per-target/per-class regime).

---

## Idea 12: Adversarial Validation via Order-Flow Causality [CRAZY]

**Concept (Execution detection, p. 5-6):** Observable order placement patterns are front-runnable. Reverse: if we assume adversaries predict our order flow, test our model's robustness to adversarial perturbations of volume/price features.

**Analogy:** Add adversarial training: perturb volume levels and trade flows to maximize WP loss; GRU learns to be robust to such perturbations. Simulate a "smart adversary" that slightly distorts the order book to fool our model. WP loss under adversarial perturbation = generalization gap simulator.

**Hypothesis:** Adversarial robustness training (ε-perturbation budget per feature) closes the valid/test WP gap (0.022 divergence noted in brief). Robustness reduces overfitting. Test WP improves by +0.005-0.015 (absolute).

**Cheapest test:**
- Use FGSM or PGD-style adversarial training on volume/price features.
- Perturbation budget ε ~ 0.05 (5% of feature std).
- Co-train: standard loss + α * adversarial loss.
- Compare test WP and valid/test divergence.

**Falsification:** Adversarial perturbation is **out-of-distribution** (unrealistic order-book moves). Model learns to ignore valid features. WP **decreases**. Or adversarial budget is **too small** to matter (no effect).

**Branch:** E2 (propensity-gated compute) or F2 (simulate LOB, pretrain on synthetic).

---

## Summary Table

| # | Idea | Concept Source | Cheapest Test | Expected Gain | Falsification | Priority |
|---|------|---|---|---|---|---|
| 1 | VW Micro-Price | VWAP (p. 3) | Compute + retrain GRU | +0.005-0.010 | Loss outweighs smoothing | HIGH |
| 2 | ACF Loss Weighting | AMIM (p. 92) | Rolling ACF + weight | +0.008-0.012 | Noise in ACF | HIGH |
| 3 | Efficiency Regimes | AMH (p. 56) | Cluster + mask | +0.006-0.015 | Static labels | MEDIUM |
| 4 | Decay Momentum | VWAP recurrence (p. 4) | Add feature | +0.003-0.008 | Redundant w/ trades | MEDIUM |
| 5 | Settlement Target | VWAP avg (p. 3) | Smooth targets + retrain | +0.004-0.010 | Info loss | MEDIUM |
| 6 | Regularity Aux Head | Detectability (p. 5) | Compute + aux loss | +0.005-0.012 | Uncorrelated w/ noise | MEDIUM |
| 7 | MII Confidence Gate | AMIM adjustment (p. 99) | Compute MII + modulate | +0.008-0.015 | Label leakage | MEDIUM |
| 8 | Anti-Detectability Reg | Exec randomness (p. 6) | ACF regularizer | +0.010-0.020 | Orthogonal to dynamics | CRAZY |
| 9 | Multi-Horizon Pyramid | VWAP horizons (p. 5) | 6 heads + blend | +0.012-0.025 | Collinear targets | HIGH |
| 10 | Per-Class Asymmetry | Efficiency variance (p. 410) | 4 models + route | +0.025-0.050 class-3 | Overfitting | HIGH |
| 11 | Variance Ratio Co-Pred | VR metric (p. 440) | VR feature + aux | +0.010-0.020 | Already captured | MEDIUM |
| 12 | Adversarial Robustness | Front-run robustness (p. 6) | FGSM training | +0.005-0.015 | OOD perturbations | CRAZY |

---

## Next Steps (for team)

- **Quick wins (1-2 sessions):** Ideas 1, 2, 10. Lowest compute cost, highest expected gain.
- **Exploratory:** Ideas 3, 5, 9. Require careful tuning but solid theory.
- **Risky/high-reward:** Ideas 7, 8, 12. Conceptually ambitious; falsification likely but potential +0.02-0.03 if they work.
- **Symreg pipeline (A4):** Ideas 7, 11 could feed symbolic regression (src/wnn/symreg.py) to find closed-form efficiency/VR relationships.
