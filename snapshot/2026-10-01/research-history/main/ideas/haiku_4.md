# Idea Generation: Transfer from CIMAWA & CEX-DEX Arbitrage

**Source documents:**
1. CIMAWA (Concept for Imitating Mental Ability of Word Association) — word association via symmetric/asymmetric strength
2. CEX-DEX Arbitrage paper (2026) — execution uncertainty, risk-averse decision-making, fallback logic, variance minimization

---

## Idea 1: Execution-Uncertainty Filtering (Risk Aversion Lens)
**Concept:** CEX-DEX paper frames arbitrage as risk-averse entry decisions: agents only attempt trades where E[profit] - λ·√Var[profit] ≥ threshold (Eq. 1, p.8). Entry is selective, not mechanical.

**Analogy to our task:** Our 9.4% scoring mask and class-3 label rarity are like execution uncertainty. We should predict only when confidence-adjusted expected gain exceeds a threshold; skip low-confidence predictions before scoring.

**Hypothesis:** A model that gates predictions based on a learned confidence threshold will improve WP by reducing wrong-direction calls in high-variance regions (class 3).

**Cheapest test:** Train a gating classifier p_gate(x) = is_confident(x) on train features. Multiply predictions by Bernoulli(p_gate(x)) or use soft gating with α·pred + (1-α)·baseline. Eval on valid_targets.npz with weight-Pearson metric.

**Falsification:** If gating reduces WP (fewer points attempted hurts global metric more than mispredictions help), the idea fails.

**Research tree branch:** C2 (diverse members) or F1 (long shots) — novel entry selection mechanism.

---

## Idea 2: Fallback Logic Decision Trees (Hierarchical Adaptation)
**Concept:** CEX-DEX paper uses state-dependent value functions with three fallback actions after failure: close immediately, retry, or wait (Fig. 3, Eq. 2, p.9). Agent recursively chooses best action at each state.

**Analogy to our task:** When is_scored is false (DEX leg fails), we have open position. We could train a fallback classifier: if primary GRU is uncertain, do one of: (a) use baseline, (b) use simpler heuristic, (c) defer to i1 cross-instrument signal.

**Hypothesis:** A hierarchical decision tree with fallback heads will beat single-path models because it can recover from primary model failure via orthogonal signals.

**Cheapest test:** Train three heads: primary (GRU), fallback_baseline (frozen baseline readout), fallback_heuristic (cache/heur_valid.npz features). Gating learned on clean train data. Score on valid.

**Falsification:** If fallback ensemble is too expensive (>80 µs/row) or underperforms simple averaging, the idea fails.

**Research tree branch:** C3 (ensemble blending) or C2 (diverse members).

---

## Idea 3: Asymmetric Association Structure for Cross-Instrument Signals (CIMAWA Lens)
**Concept:** CIMAWA combines symmetric (PMI: A→B relation strength = B→A) and asymmetric (Walter-Rapp: A→B ≠ B→A) word associations (p.10-11, Eq. 1). Asymmetry captures directionality.

**Analogy to our task:** i0 and i1 may have asymmetric information flow. i0 leads to i0's targets, but i1 may lead-lag (forward or backward). Model this asymmetry explicitly: a "forward association" head (i0→t0) and "reverse association" head (i1→t0 predicting i0 moves).

**Hypothesis:** Separating forward (i0→i0) and reverse (i1→i0) association heads, then blending with learned asymmetry weights, will improve class-3 WP where i1 carries hidden side information.

**Cheapest test:** Train two attention heads: one over i0 features only, one over i1 features. Learn weights α_fwd, α_rev via meta-learning on valid. Eval W.P. on heur_valid.npz.

**Falsification:** If asymmetric heads are redundant with standard attention (both learn same patterns), idea fails.

**Research tree branch:** B3 (i1 cross-instrument) or A4 (symbolic explanations via association strength).

---

## Idea 4: Regret-Minimized Loss for Class-3 (CEX-DEX Regret)
**Concept:** CEX-DEX paper models agent regret from two sources: (1) successful profits foregone (π^s - θ), (2) losses from failed executions (π^f). The agent minimizes total regret, not just loss variance (p.7-8).

**Analogy to our task:** Class-3 has high WP variance (0.42-0.44) and is 46% of metric weight. Wrong direction in class-3 is costlier than wrong magnitude elsewhere. Define regret as: cost_of_wrong_sign × loss + weighted_variance_penalty.

**Hypothesis:** A loss that explicitly penalizes direction error in class-3 more heavily, weighted by target magnitude, will push class-3 WP from 0.42 to 0.50+.

**Cheapest test:** Retrain GRU (existing setup) with loss = W1·sign_loss(class=3) + W2·MSE(all classes). Tune W1 on valid_clean. Score via gmetric.py.

**Falsification:** If class-3 sign accuracy improves but overall metric decays (classes 0-2 suffer), tradeoff is unfavorable.

**Research tree branch:** C1 (GRU recipe tuning) — loss modification.

---

## Idea 5: Time-Adaptive Slot Encoding (CEX-DEX Subslot Regime Switching)
**Concept:** CEX-DEX paper compares 12-second and 1-second "slot" regimes, and price interpolation differs per regime (p.4-5, Fig. 1). Price dynamics depend on execution rhythm: steady vs bursty.

**Analogy to our task:** Order-book rows arrive in bursts during high activity and sparse during quiet periods. Encoding should adapt: 50-row window for burst detection, 2-row window for sparse regimes. Slot = "batch of rows with coherent microstructure".

**Hypothesis:** An adaptive RNN that detects regime (busy/quiet) and reshapes feature aggregation windows will better predict price moves because the relevant time scale is endogenous.

**Cheapest test:** Train regime classifier from inter-row spacing or volume. Build dual-window GRU (fast and slow paths) with regime-gated attention. Eval on valid.

**Falsification:** If overhead (per-row regime switching) > gain in WP, or if regime is not predictive, idea fails.

**Research tree branch:** B1 (heuristic library) — regime detection as input signal.

---

## Idea 6: Variance-Constrained Target Loss (CEX-DEX Variance Minimization)
**Concept:** CEX-DEX paper emphasizes that faster execution reduces variance of both wins and losses, making risk-averse agents more willing to trade (p.2-3, p.23). The agent minimizes E[·] - λ√Var[·], not just E[·].

**Analogy to our task:** Our WP metric is global and pooled across all rows. Sequences with high target variance (class 3) drag down the metric. If we train to minimize heteroskedasticity by target magnitude, we reduce variance in high-|y| regions.

**Hypothesis:** Add a variance regularization term to loss: λ_var · E[|y - ŷ|^2 · |y|] to penalize large errors on large targets.

**Cheapest test:** Retrain GRU with loss = MSE + λ_var·weighted_variance_loss. Tune λ_var on valid. Eval on valid_clean with gmetric.

**Falsification:** If variance penalty forces underfitting (worse average loss without WP gain), idea fails.

**Research tree branch:** C1 (GRU recipe tuning).

---

## Idea 7: State-Dependent Regime Heads (CEX-DEX Value Functions by State)
**Concept:** CEX-DEX fallback logic solves state-dependent value functions: V^f(t, k) where k = number of failed attempts. Each state has a different optimal policy (p.9, Eq. 2). Solver uses backward induction per state.

**Analogy to our task:** Market regimes (quiet, transition, busy) likely have different label generation processes. Instead of one GRU, train separate heads per regime, each with its own value function (target distribution). Blend via gated attention.

**Hypothesis:** A mixture-of-experts model with regime-specific heads (quiet_expert, transition_expert, busy_expert) will outperform single GRU because label mechanics differ per regime.

**Cheapest test:** Cluster train seqs by inter-row spacing / volume into 3 regimes. Train three GRU heads, one per regime. Learn gating on valid. Eval on valid_targets.npz.

**Falsification:** If head capacity is wasted on regime-specific overfitting (gains vanish on test), idea fails.

**Research tree branch:** C2 (diverse members) or F2 (treat each sequence as new instrument).

---

## Idea 8: Execution Probability Weighting & Margin-to-Boundary (Sophisticated Importance Weighting)
**Concept:** CEX-DEX paper introduces execution probability α (p.5-6): if α is high, agent is bold; if low, agent only attempts high-margin trades. We should weight samples by both α and margin to decision boundary.

**Analogy to our task:** is_scored is binary but predictable (AUC ~0.9 from features). Weight training samples by: w = p(scored|x) · margin(pred to sign flip). High-margin correct predictions get high weight; low-margin predictions near zero get down-weighted.

**Hypothesis:** Sample importance weighting by p(scored|x)^2 · margin will reduce overfitting to boundary cases and improve WP.

**Cheapest test:** Train p_scored(x) classifier. For each train sample, compute margin = |pred - 0| / pred_std. Retrain GRU with sample_weight = p_scored^2 · margin. Eval on valid.

**Falsification:** If margin weighting introduces noise (high variance estimates of margin) or stability issues, idea fails.

**Research tree branch:** D2 (weights: label uniqueness, propensity temperature).

---

## Idea 9: CIMAWA-Style Code Association Strength (Symbolic + Asymmetric)
**Concept:** CIMAWA evaluates both symmetric (PMI) and asymmetric (Walter-Rapp) association strength between words (p.10, Eq. 1). It finds that some relations are one-directional: "Canary" → "bird" is strong, but "bird" → "Canary" is weak.

**Analogy to our task:** Price codes are anonymized but deterministic. Apply CIMAWA to sequences: for each code pair (code_i, code_j), compute association strength to "UP" vs "DOWN" labels. Find asymmetric code patterns: code_A strongly predicts UP but code_B only weakly predicts UP (one-way).

**Hypothesis:** A symbolic rule library derived from CIMAWA code associations will identify class-3 hard examples because multi-tick patterns have asymmetric predictability (some orderings favor one direction).

**Cheapest test:** Extract all 2-grams of price codes in class-3 seqs. Compute CIMAWA strength to label. Fit symbolic regression (src/wnn/symreg.py) with these association features. Eval on valid.

**Falsification:** If code associations are too noisy or dominated by single-code features (no real pair effect), idea fails.

**Research tree branch:** A4 (symbolic explanations) or A2 (decode encodings).

---

## Idea 10: Risk-Averse Ensemble Weighting by Variance Profile
**Concept:** CEX-DEX agent is risk-averse: it uses expected profit minus variance penalty (Eq. 1, p.8). Each decision path has different variance; agent weights decisions by variance profile, not uniform averaging.

**Analogy to our task:** Instead of averaging ensemble members equally, compute variance of each member on each sample. Low-variance members get higher weight on uncertain rows; high-variance members get lower weight.

**Hypothesis:** Variance-weighted ensemble blending (not uniform average) will improve WP by reducing tail risk (catastrophic wrong directions on hard rows).

**Cheapest test:** Train 3 diverse members (baseline, GRU, GBM on heuristics). For each valid sample, compute per-member variance via Bayesian dropout or MC sampler. Blend with weights ∝ 1/variance. Eval on valid_targets.npz.

**Falsification:** If variance estimates are unreliable (high noise) or high-variance members are right more often, idea fails.

**Research tree branch:** C3 (ensemble blending).

---

## Idea 11: Lookahead Regret-Bound Learning (CEX-DEX Fallback Anticipation) 🚀
**Concept:** CEX-DEX agent anticipates fallback outcomes: if DEX leg fails, what will be the cost? The agent computes expected fallback value before attempting (p.8-9, Eq. 2). This is lookahead regret.

**Analogy to our task:** Train a secondary classifier to predict: "Would this prediction be corrected if we looked ahead K rows?" Use this as sample importance. High lookahead-regret samples (correctable with future info) get lower weight; low-regret samples (already optimal) get higher weight.

**Hypothesis:** Lookahead-regret weighting will reduce overfitting to transient mispredictions and improve generalization because the model learns which errors are "correctable" by future dynamics.

**Cheapest test:** Build lookahead classifier: seq[t:t+10] predicts if seq[t]'s label is "corrected" by seq[t+20]'s actual label. Retrain GRU with w = lookahead_error (high error = low weight). Eval on valid.

**Falsification:** If lookahead classifier is too weak (AUC < 0.55) or future info is not available at test time, idea fails.

**Research tree branch:** F3 (privileged-teacher label denoising) — a variant.

---

## Idea 12: Preconfirmation Horizon for Class-3 (Execution Timing Analogy) 🚀
**Concept:** CEX-DEX paper models execution time uncertainty: agents must choose when to execute (now, retry later, wait). The confirmation slot is stochastic. Similarly, our class-3 exact tick-path rule depends on "when F is reached" (brief, ~1 book event, p.1 brief).

**Analogy to our task:** Class-3 labels follow an exact rule but the "confirmation horizon" (when F is reached) is hidden but locally constant. Instead of predicting {t0, t1} directly, predict: (a) which horizon applies (1 event, 2 events, 4 events ahead), (b) then apply exact rule for that horizon.

**Hypothesis:** A two-stage model—first predict horizon, then apply horizon-conditioned exact rule—will improve class-3 WP by making the label generation process more transparent.

**Cheapest test:** Retrain GRU to jointly predict t0, t1 and "horizon_code" (discrete: 1/2/4 events). Freeze rule head to enforce exact-law supervision. Eval on valid_clean and check if horizon correlates with book-event spacing.

**Falsification:** If horizon is not recoverable from features (constant throughout seqs), or if rule supervision is too tight, idea fails.

**Research tree branch:** B2 (class-3 exact tick-path features) or A1 (target law refinement).

---

## Summary

**8-12 ideas delivered:**
1. ✅ Execution-uncertainty filtering
2. ✅ Fallback logic decision trees
3. ✅ Asymmetric association structure (i1 signals)
4. ✅ Regret-minimized loss for class-3
5. ✅ Time-adaptive slot encoding
6. ✅ Variance-constrained target loss
7. ✅ State-dependent regime heads
8. ✅ Execution probability weighting
9. ✅ CIMAWA-style code associations
10. ✅ Risk-averse ensemble weighting
11. ✅ Lookahead regret-bound learning (🚀 unconventional)
12. ✅ Preconfirmation horizon for class-3 (🚀 unconventional)

**Unconventional ideas marked with 🚀:** 11, 12 (and arguably 5, 9 are offbeat).

All map to research/tree.md branches; all are testable on existing cache/valid_targets.npz and heuristic features.
