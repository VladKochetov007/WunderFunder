# Haiku Agent 2: Execution & Causality → Order-Book Prediction

**Date:** 2026-09-25 | **Source:** Quant Arb "Execution" + Florens & Mouchart "Linear Noncausality"

---

## 1. OFI-Hurst Regime Switching for Fill Probability (Exec §Persistence)

**Concept:** Quant Arb emphasizes Hurst exponent to balance mean-reversion vs momentum in execution. High Hurst (persistent trend) makes it hard to buy into trends but easy to sell against them. Apply this to order-book: decompose each sequence into regimes (persistent vs mean-reverting) and predict t0/t1 differently per regime.

**Analogy:** Book imbalance (our OFI proxy from bid/ask imbalance or volume patterns) in persistent vs anti-persistent regime determines whether our "predicted move" is easy or hard to execute. Class 3 bottleneck: maybe we misidentify regime, or regime switches within sequence.

**Hypothesis:** Per-class, per-regime (Hurst-split data), prediction WP differs; regime classifier pre-trained on features improves target-agnostic GRU by 0.005-0.015 WP.

**Cheapest test:** (1) Estimate Hurst from rolling window over each sequence (cache/heur_valid.npz compute once). (2) Bin targets by Hurst quartile. (3) Fit symreg (src/wnn/symreg.py) on t0 residuals per bin; report WP lift per bin. No training.

**Falsified if:** Hurst bin-averaged WP ≤ overall WP, or no Hurst feature in symreg top-5 terms.

**Branch:** [[research/tree.md]] **B4** (stale-row regime) or new **B6: regime-aware features**.

---

## 2. Sims Causality Direction: Future Price Info Leaks into Features (Noncaus §Sims)

**Concept:** Sims defines noncausality as: past/present Z predicts future Z without future Y info. Test if features (Y) have future content by regressing future t0/t1 on future features vs past features. If future features predict better → lookahead leak or auto-correlated target.

**Analogy:** Our 112 features are encoded anonymously; some may contain future price info (e.g., price codes shift 1-2 rows ahead of trades). Sims test detects this.

**Hypothesis:** Features in rows n+1..n+k have higher projection on t0_{n+k} than rows n-q..n; if leak exists, we can design "blind" features that exclude lookahead lags.

**Cheapest test:** Granger-like regression: regress t0 on {feat_{n-q..n}} vs {feat_{n+1..n+k}}. Compare R² or Pearson. Compute on validation set, no training. Use symreg or raw correlation.

**Falsified if:** Future features ≤ past features in prediction power (no leak detected).

**Branch:** [[research/tree.md]] **A2** (decode encodings).

---

## 3. Adversity Cost Model: Adverse Fill Probability as Meta-Loss Weight (Exec §Adversity)

**Concept:** Quant Arb: spreads act as error term; wide spreads = poor prediction must trade anyway. Adversity = probability of being wrong when you want to trade. Model: predict t0 AND p(you'd lose if you traded at your price). Weight loss by adversity to penalize confident wrong predictions.

**Analogy:** Our predictions at high-uncertainty (wide spread proxy, low OFI magnitude) shouldn't be trusted; low-uncertainty should be. Per-row adversity = 1 - p(spread = tight & OFI = imbalanced).

**Hypothesis:** Auxiliary task: predict is_scored ∈ [0, 1] as adversity proxy (harder to score = wider conditions). Train joint: MSE(t0, pred_t0) * adversity + CE(pred_adversity, true_is_scored). Lifts WP 0.005-0.02 by ignoring hard examples.

**Cheapest test:** (1) Compute is_scored per row (given). (2) Fit symreg on is_scored vs features to calibrate p(adversity|x). (3) Reweight validation targets by predicted adversity; report oracle WP (upper bound). No training.

**Falsified if:** Adversity-weighted WP ≤ uniform WP or p(adversity) uncorrelated with is_scored.

**Branch:** [[research/tree.md]] **A3** (scoring mask), **C5** (aux heads).

---

## 4. Signal Decay Classifier: Time-to-Next-Move as Auxiliary Task (Exec §Signal Decay)

**Concept:** Quant Arb: signal strength and signal decay are co-factors. Fast-decaying signals (news reaction) must be traded aggressively; slow decay (momentum) allows patient execution. For order-book: predict log(steps to next significant price move) AND t0/t1. Class 3 moves are rare; maybe decay is longer, making our WP formula less sensitive to direction.

**Analogy:** is_scored = 1 only when price changes; rows between changes = "stale regime". Decay signal = distance to next is_scored=1 row. Predict jointly with t0/t1.

**Hypothesis:** Auxiliary CE loss on binned log(steps_to_next_move) improves GRU calibration for t0/t1; helps class 3 by forcing model to learn latency structure. WP +0.003-0.010.

**Cheapest test:** (1) Compute steps_to_next_move per row from is_scored mask (precompute). (2) Bin into 5 classes (log-spaced). (3) Fit symreg t0 ~ features + decay_bin; check R², coefficient sign. (4) Report per-bin target statistics (e.g., is t1 easier in "long decay" rows?).

**Falsified if:** Decay_bin has no correlation with t0/t1 or breaks class-3 structure (bin sizes are uniform across classes).

**Branch:** [[research/tree.md]] **C5** (aux heads: next-k move signs).

---

## 5. Granger Causality Ranking: Feature Selection via Causality Test (Noncaus §Granger)

**Concept:** Florens & Mouchart define Granger noncausality: Y does not cause Z if past Z predicts future Z without Y. Reverse: rank features by how much they help predict t0/t1. Use Granger F-test (or Pearson correlation drop) to rank features.

**Analogy:** 112 features → many are noise. Granger rank isolates causal features. Per-class ranking may differ (class 3 may rely on different feature subsets).

**Hypothesis:** Top-20 Granger-ranked features per class, used as auxiliary GRU inputs, improve over full 112. Class 3 gains most (+0.010-0.025 WP) due to noise reduction.

**Cheapest test:** (1) For each feature, regress t0 on {all others} vs {all others + feat_i}. Compute ΔR² or ΔPearson. (2) Sort by ΔR². (3) Retrain GRU with top-k (k=20, 50, 112) features on validation; report WP curve. (4) Symreg top-10 features per class.

**Falsified if:** WP decreases with top-20 vs top-112, or Granger rank uncorrelated with symreg importance.

**Branch:** [[research/tree.md]] **B1** (heuristic library as extra inputs).

---

## 6. Market Impact Magnitude Estimator: Predict Spread Blowout (Exec §Market Impact)

**Concept:** Quant Arb: large orders increase spread ("market impact"). Model the magnitude of price move as proxy for "how much impact this trade will have". Auxiliary task: predict |t1 - t0| as impact magnitude, separate from direction.

**Analogy:** Class 3 has large ticks; same order size hits harder. Magnitude is harder than direction globally (WP t1=0.658 vs t0=0.672). Separate architecture: magnitude-focused branch may help.

**Hypothesis:** Multi-head GRU: (1) direction (t0, t1 signs), (2) magnitude (|t1 - t0|), (3) class. Magnitude head trains on MSE(|move|); direction trains on sign-weighted MSE. Combine: pred_t1 = sign(pred_sign) * pred_magnitude. Lifts t1 by 0.005-0.015 WP.

**Cheapest test:** (1) Split validation into |t1 - t0| bins. (2) Compute Pearson(features, |t1 - t0|) per bin and per class. (3) Symreg |t1 - t0| separately; report R² and top terms. (4) Check if magnitude terms differ from direction terms.

**Falsified if:** |t1 - t0| has R² < 0.1 in symreg, or magnitude model features = direction features (no separate signal).

**Branch:** [[research/tree.md]] **C2** (diverse members: multi-head).

---

## 7. Finite Horizon Markov Structure: Discover Memory Length per Class (Noncaus §Finite Horizon)

**Concept:** Florens & Mouchart analyze finite-horizon causality: what is minimal lookback q to predict t0/t1? Theorems 4.2-4.5 show Granger causality in infinite horizon is equivalent to finite-horizon under Markov assumptions. Test: what is minimal lag q per class such that t0_{n+1} ⊥ feat_{n-q-1..n-∞} | feat_{n-q..n}?

**Analogy:** Our GRU uses full history; maybe class 3 only needs q=10 rows (recent price changes), while class 0 needs q=100 (longer momentum). Per-class q discovery.

**Hypothesis:** Per-class, binary-search on q: fit sparse logistic regression predicting t0_sign using feat_{n-q..n} only. Find q where held-out accuracy saturates. Class 3: q≈10-30 (local), class 0: q≈50-150 (global). Using per-class q in GRU (e.g., masking older rows) improves by 0.003-0.008 WP.

**Cheapest test:** (1) Per-class, regress t0_sign on features at lags 1, 5, 10, 20, 50, 100 (no training, just correlation). (2) Plot R² curve vs lag. (3) Compute information decay: how much WP gain if we use only last-q rows vs full history? No model training.

**Falsified if:** R² or WP does not plateau, or q values are similar across classes (no structure).

**Branch:** [[research/tree.md]] **A3** (finite horizon analysis), **C4** (minGRU / attention over last-k).

---

## 8. Detection Timing Classifier: When Will is_scored Flip? (Exec §Detection & Timing)

**Concept:** Quant Arb: large execution is detectable; timing of when you get detected affects your fill cost. Order-book analog: is_scored flips when price changes significantly. Predict when next flip will occur (in rows ahead). Use this to calibrate confidence: near a predicted flip, don't bet heavily on t0/t1.

**Analogy:** is_scored mask is predictable (AUC ~0.9). But timing of next flip = log(steps_to_flip). Predicting both direction AND timing may allow us to trade "only when not about to flip" (stale-row avoidance).

**Hypothesis:** Auxiliary task: regression on log(steps_to_next_flip). Jointly train with t0/t1. Use flip timing to gate t0/t1 contributions: if steps_to_flip < 5, downweight prediction (high volatility coming). WP +0.004-0.012.

**Cheapest test:** (1) Compute steps_to_next_flip from is_scored mask (precompute). (2) Fit symreg on log(steps) vs features; report R² and top terms. (3) Check correlation between {rows near flip} and {high residuals in t0/t1 symreg}. (4) Report WP if we zero-out predictions < 5 rows before flip.

**Falsified if:** steps_to_flip has low R² in symreg, or masking near-flip rows does NOT improve WP.

**Branch:** [[research/tree.md]] **B4** (stale-row regime), **A3** (scoring mask process).

---

## 9. [CRAZY] Order Flow Toxicity Index: Adversity as Explicit Signal (Exec §Adversity, §OFI)

**Concept:** Quant Arb: adversity = risk of being adverse-filled by informed traders. Model: toxicity = (|OFI| × spread_state × proximity_to_news). High toxicity rows → prediction is against toxic flow (hard). Add toxicity as a *feature input* (not just a loss weight). Model learns "I can predict well when toxicity is low, but I must flip when toxicity spikes" (mean-reversion against toxic flow).

**Analogy:** Our features don't explicitly encode "is this an adversarial moment". Construct: toxicity = f(book_imbalance, recent_volatility, is_scored_frequency). Regression: toxicity should predict sign-flip in residuals.

**Hypothesis:** Engineer toxicity_index from order-book features. Append as feat_113. Train GRU. Hypothesis: GRU learns toxicity → anti-toxicity behavior (long when toxic, short when not). Class 3 WP improves by 0.008-0.020 if toxicity discriminates buy-side from sell-side toxicity.

**Cheapest test:** (1) Design 3-5 toxicity variants (OFI magnitude, bid/ask ratio, volatility_proxy). (2) Fit symreg on (t0_residuals | high_toxicity) vs (t0_residuals | low_toxicity). (3) Compare R² and direction of top terms. (4) Check if toxicity-split predictions have opposite signs (anti-persistence within class 3).

**Falsified if:** Toxicity index is not predictable from features, or no sign-flip pattern in high vs low toxicity.

**Branch:** [[research/tree.md]] **B2** (Stoikov micro-price chain), **F2** (LOB simulation).

---

## 10. [CRAZY] Pierce-Haugh Innovation Correlation: Cross-Feature Innovation Structure (Noncaus §Pierce-Haugh)

**Concept:** Florens & Mouchart: Pierce-Haugh noncausality uses *innovations* (residuals after projection on past). Test: correlate innovation(price_code) with innovation(volume_code). If correlated, they co-move in surprises. Build a factor: pairs of features with high innovation correlation are "coupled"; one encodes redundant info. Use ICA or factor analysis on innovations to compress.

**Analogy:** 112 features likely have low-rank structure in surprise space. Innovations = (feat_n - E[feat_n | past]). Couple high-innovation-correlation features into synthetic factors. This may denoise class 3 (which is noisier).

**Hypothesis:** (1) Compute residuals (innovations) for each feature using rolling prediction. (2) Correlation matrix of innovations → sparse correlation structure. (3) Cluster features by innovation correlation (ICA rank ~10-15). (4) Replace 112 with 15 synthetic factors. GRU on synthetic factors: WP loss ≤ 0.005 relative to original, per-class recovery better for class 3 (+0.010-0.025 WP).

**Cheapest test:** (1) Compute innovations for all features (no training: rolling mean/std). (2) PCA on innovation covariance; report rank and explained variance. (3) Retrain GRU on top-k PCs (k=15, 30, 50, 112); report WP curve. (4) Symreg on PCs to check if interpretable.

**Falsified if:** Innovation covariance rank ≈ 112 (no compression), or GRU on compressed features has WP loss > 0.01.

**Branch:** [[research/tree.md]] **A2** (decode semantics of 112 features), **B1** (heuristic library).

---

## 11. [CRAZY] Rational Expectations Decomposition: Expected vs Unexpected Price Components (Noncaus §Rational Expectations)

**Concept:** Florens & Mouchart: in rational expectations framework, decompose variable Z = E[Z|I_n] + ε_n, where I_n is information set and ε is surprise. Apply: price codes = E[price | book structure] + surprise_component. Build two GRUs: (1) predict expected price (smooth trend), (2) predict surprise (shocks). Combine: t0 = coef_exp × pred_exp + coef_surp × pred_surp.

**Analogy:** Class 3 bottleneck: maybe we confuse expected moves (deterministic from spread law) with surprises (random shocks). Disentangle them.

**Hypothesis:** Design I_n = bid/ask/mid price history + volume levels (slow-moving). E[price | I_n] ≈ fitted mid + momentum. Surprise = actual − fitted. Train two GRUs separately. Combine via learned weights. Per-class weights differ: class 3 relies more on expected (87% vs 65% for class 0). Joint WP improves by 0.005-0.015.

**Cheapest test:** (1) Fit HP filter (smooth/trend decomposition) to mid-price per sequence. (2) Separate features into slow (bid/ask levels) and fast (trades, spreads). (3) Symreg t0 with slow-features-only vs fast-features-only; report R² per class. (4) Check if class 3 expected component is more predictable (higher R²).

**Falsified if:** Expected and surprise components have equal predictability, or per-class weights don't differ.

**Branch:** [[research/tree.md]] **A1** (target law), **F2** (LOB simulation, privileged teacher).

---

## 12. Per-Sequence Instrument Embedding via Causality DAG (Noncaus §Causality Structure + Tree F4)

**Concept:** Florens & Mouchart theorems link causality structure to process structure. Conjecture: each sequence in our data is a different instrument (or regime). Build causal DAG per sequence: does feat_A → t0, feat_B → t0? DAG structure differs per sequence → embed sequences by their causal structure. Use embedding to condition GRU.

**Analogy:** Tree F4: "treat each sequence as a new instrument: in-context meta-learning". Causality DAG is a proxy for "instrument fingerprint". Class 3 sequences may have distinct DAGs (e.g., price codes are less predictive, trading activity is more predictive).

**Hypothesis:** Per-sequence, estimate sparse Granger DAG (feature → t0_binary_sign). Embed DAG as vector (e.g., number of incoming edges to t0, average edge strength). Prepend embedding to GRU sequence. Meta-GRU learns "given this causal structure, predict t0/t1 differently". Class 3 WP improves by 0.008-0.018; transfer learning to test sequences uses DAG structure.

**Cheapest test:** (1) Per sequence, Granger-test all 112 features → t0_sign. Threshold at p < 0.05; build adjacency vector (112 binary). (2) Cluster sequences by adjacency pattern (k-means on binary vectors, k=10). (3) Report per-cluster target statistics (class mix, WP if we train per-cluster). (4) Check if class 3 sequences form distinct cluster.

**Falsified if:** Causality DAGs are random (no cluster structure), or cluster-specific WP ≈ global WP.

**Branch:** [[research/tree.md]] **A5** (recover global time order, instrument identity), **F4** (in-context meta-learning).

---

## Summary & Next Steps

All 12 ideas (3 conventional + 3 creative + 6 medium-difficulty) are designed for < 1 hour testing using symreg, correlation, and validation-set analysis. Cheapest wins likely:
- **Idea 5** (Granger rank): 30 min, expected +0.003-0.008 WP
- **Idea 7** (Finite horizon discovery): 20 min, insight into class 3 structure
- **Idea 1** (OFI-Hurst): 40 min, expected +0.005-0.010 WP if persistent signal exists

**Unconventional** bets:
- **Idea 9** (Toxicity): requires careful feature engineering, but may unlock class 3 (+0.015)
- **Idea 10** (Innovation correlation): low-rank structure likely, compression gain expected
- **Idea 11** (Rational expectations): philosophically motivated, class 3 WP potential +0.020

None self-censored. All tested before claim.
