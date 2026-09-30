# Round 2 Ideas — Agent D (Haiku)
## Robustness, denoising, ensemble logic

---

### 1. **Elliptical Wishart Covariance Decoder for Test Shift**
**(a) Concept:** Use robust covariance estimation (elliptical Wishart, t-Wishart for heavy tails) on valid features to recover a covariance structure that resists distribution shift. The score matrix residuals from PCA decomposition isolate the non-Gaussian tail behavior.  
**Source:** Ayadi et al. "Elliptical Wishart Distribution" (heavy-tailed covariance on manifolds).

**(b) Analogy to data:** Our feature space is rank-Gauss transformed with sentinel missing values. If test rows have fatter tails or extreme book states (quiet regime, flash crash, etc.), standard covariance breaks. Elliptical models generalize Wishart to non-Gaussian noise.

**(c) Testable hypothesis:** A model trained on features whitened by t-Wishart covariance (fit on valid) will transfer better to test than Gaussian PCA whitening, especially in class 3 (large tick, sparse moves).

**(d) Cheapest test:** Compute full-rank covariance on valid features; fit t-Wishart via Riemannian optimization (already in src/wnn/symreg.py design pattern); whiten validation rows using `Σ^{-1/2}` from the fit; measure softened-mask WP lift. (~5 min, no training, uses cache/heur_valid.npz).

**(e) Falsifier:** If softened-mask WP drops >0.001 vs baseline, or is unchanged, the residual structure from elliptical tails is not predictive of test. The sentinel encoding may already capture tail behavior.

**(f) Research tree:** A3 (Scoring mask process) + C4 (alternative architectures); branch: "adaptive whitening by tail regime."

---

### 2. **Micro-Price Ladder: Tick-Space PCA on Class-3 Deltas**
**(a) Concept:** Apply PCA residuals directly in tick space, not price space. Decompose the *change* in target (t0, t1 deltas row-to-row) by their tick indices; the first PC should isolate the "dominant tick move," second PC the "bid-ask imbalance move."  
**Source:** PCA Unleashed; tick-space is a discrete geometry (like tenor-space residuals in EUR swaps).

**(b) Analogy to data:** Our class-3 target changes only when bid/ask codes change. Between changes, target is constant. This makes deltas *sparse and discrete*. PCA on the delta distribution (not levels) naturally finds the dominant tick direction—analogous to slope/curvature in curves.

**(c) Testable hypothesis:** PCA residuals of t0/t1 deltas will explain ~80% of variance with one PC (the "principal tick direction"); AUC of next-move classification improves from 0.74 to 0.78+ if we use this PC + the 3-way branch rule.

**(d) Cheapest test:** Extract t0/t1 delta time series within class 3 (valid); run PCA on deltas only (not targets); project deltas onto first 2 PCs; use PC scores + branch as features in scripts/lift_probe.py. (~2 min, no training).

**(e) Falsifier:** If PC1 doesn't correlate with current bid/ask imbalance (e.g., VB > VA) or if residuals are still highly non-stationary, tick-space PCA is too naive. The targets are a function of *future* book state, not deltas.

**(f) Research tree:** A1 (Target law, class 3 exact) → A4 (symbolic explanations); branch: "tick-ladder manifold."

---

### 3. **Label-Denoising via Privileged-Teacher Cross-Instrument Lookahead** *(unconventional)*
**(a) Concept:** Use i1 (cross-instrument) future data as a privileged teacher signal. For each sequence, fit a linear model to predict i0's target change *from i1's book state* at the same time. Residual label noise is the gap between i0's true label and what i1's signal predicts. Retrain GRU with instance weights ∝ `1 / (1 + noise_estimate)`.  
**Source:** Privileged-info learning (Vapnik), label denoising (confident learning paradigm).

**(b) Analogy to data:** If i0 and i1 are related instruments (e.g., paired ETFs, futures vs spot), their microstructure should be correlated. i1's bid-ask ladder (a0..a7) is exogenous info that leaks i0's future state. This is *not* cheating—the organizer's baseline can't use i1 signals because it's pre-fitted.

**(c) Testable hypothesis:** GRU trained with cross-instrument label confidence weights (soft targets = original + (1-conf)*i1_prediction) will improve class 3 WP by 0.005-0.010 (softened mask).

**(d) Cheapest test:** Build a closed-form regressor: `t0_change[i] ~ a0[i] + a1[i] + trade_flow_i1[i]` (symreg on 3 terms); compute residual per row as noise estimate; apply to GRU training as class weight in scripts/train_rnn.py (one epoch seed test). (~3 min).

**(e) Falsifier:** If noise estimates are anti-correlated with actual label correctness (e.g., high-noise rows are still easy), the i1 signal is not informative. Test: measure corr(confidence, |target_error|) on holdout.

**(f) Research tree:** B3 (i1 cross-instrument lead-lag) → D2 (mask-softening weights); branch: "exogenous pair-denoising."

---

### 4. **Stale-Row Regime Detection via Local Recency Eigenvalues**
**(a) Concept:** Compute eigenvalues of the covariance matrix of recent 20 rows vs old 50 rows separately (locally, per sequence). When eigen-spectrum flattens (condition number drops), rows are "stale" (feature repeatability is high, moves are rare). Use this as a binary stale/active regime predictor; train separate micro-GRUs for each regime.  
**Source:** RMT (Random Matrix Theory), eigenvalue drift as regime detector (noted in research/tree.md H1-8).

**(b) Analogy to data:** Class 3 has a "quiet after moves" regime where WP is 0.36 after >170 silent rows (B4 in tree). Eigenvalues naturally detect this: when prices don't change, feature variance collapses. This is a *free* signal from the feature covariance itself.

**(c) Testable hypothesis:** Rows with condition number(cov_recent) > 10 belong to "active regime" (AUC 0.80); rows with CN < 3 are "stale" (WP 0.36 baseline). Training separate heads for each regime yields +0.004-0.006 WP (softened).

**(d) Cheapest test:** For each sequence in valid, compute rolling (t-20, t) and (t-50, t-20) covariance matrices; eigendecompose; compute condition numbers; bin rows; run scripts/decide.py --lam 0.02 with regime-stratified losses. (~4 min, no training).

**(e) Falsifier:** If condition number doesn't correlate with move frequency (e.g., quiet rows have high CN anyway), the eigenvalue drift is decoupled from the target predictability. Check: corr(CN, "next 5 rows contain a move").

**(f) Research tree:** B4 (stale-row regime) → C2 (diverse members); branch: "spectral regime filter."

---

### 5. **Blend Orthogonalization: GRU + Baseline via Gram-Schmidt Residuals**
**(a) Concept:** Current blend is 0.5 GRU + 0.5 baseline. Instead, orthogonalize: fit baseline first on valid; compute residuals; fit GRU on residuals only; blend as baseline + α × GRU_residual where α is tuned on valid. This ensures ensemble members are statistically independent.  
**Source:** Gram-Schmidt process (numerical LA); Frisch-Waugh-Lovell (econometrics).

**(b) Analogy to data:** The baseline captures "level" (PC1-like behavior: mean price direction). GRU should capture "higher-order" (slope/curvature, branch direction, timing). By orthogonalizing, we avoid double-counting the baseline's easy wins.

**(c) Testable hypothesis:** Orthogonalized blend (baseline + β·GRU_resid) yields higher softened-mask WP (0.659+) than 0.5-0.5 blend (0.6569) because member redundancy is eliminated. Optimal β ∈ [0.8, 1.5].

**(d) Cheapest test:** Cache baseline predictions on valid; compute residuals; fit GRU on residuals only (one epoch, quick); test blend on valid with grid search over α ∈ [0, 1.5]; measure softened-mask WP. (~2 min, 1-epoch GRU).

**(e) Falsifier:** If orthogonalized blend underperforms simple 0.5-0.5, the correlation between baseline and GRU predictions is actually *beneficial* (perhaps both agree on easy rows, disagree on hard ones). Measure: correlation of baseline and GRU preds per sequence.

**(f) Research tree:** C3 (distill baseline + GRU blend) → D2 (ensemble weighting); branch: "orthogonal stack."

---

### 6. **Adaptive Softening Lambda: Per-Sequence Mask Entropy** *(unconventional)*
**(a) Concept:** The softened mask (lambda=0.02) is global. But different sequences have different mask densities (some may be 5% scored, others 15%). Fit a per-sequence lambda via entropy: `lambda_seq = 0.02 * (1 - entropy(is_scored) / log(2))`. Sequences with uniform masks get lambda=0.02; sequences with clustered scoring (e.g., only near moves) get lambda=0.01.  
**Source:** Maximum entropy principle; adaptive regularization.

**(b) Analogy to data:** Test mask may be softer than valid *on average*, but some sequences have harder masks (rare scored rows) and need less softening. Entropy captures this naturally without hand-tuning.

**(c) Testable hypothesis:** Per-sequence adaptive lambda (fit on valid, apply to softened-mask metric) correlates better with LB than global lambda=0.02. Expected: Spearman rank corr with LB > 0.75 (vs ~0.70 now).

**(d) Cheapest test:** Compute is_scored entropy per sequence in valid; fit lambda_seq as above; recompute softened-mask metric with variable lambda; rank models; measure corr with LB model ranks. (~1 min, no training).

**(e) Falsifier:** If per-sequence lambda is uncorrelated with model ranking, the mask softness is truly global and doesn't vary by sequence. The entropy-based adjustment is noise.

**(f) Research tree:** A3 (Scoring mask process) → D1 (decision rule); branch: "entropy-adaptive softening."

---

### 7. **Class-Conditional Symbolic Regression: Multi-Tree Mixture**
**(a) Concept:** Run SISSO (src/wnn/symreg.py) separately on each class (0, 1, 2, 3) residuals. Each class gets its own symbolic regression tree (e.g., `t0_resid ~ min(vol_imbalance, threshold) - tanh(trade_EMA)`). Blend GRU predictions by interpolating per-class trees: `pred_final = GRU + blend_weight(class) * symreg_tree[class](features)`.  
**Source:** SISSO engine (already in codebase); mixture of experts (MoE) paradigm.

**(b) Analogy to data:** We already know class 3 responds to different features than classes 0-2 (different tick sizes, different move frequencies). Symbolic regression amplifies this: each class can discover its own interpretable correction rule.

**(c) Testable hypothesis:** Class-conditional symreg corrections yield +0.004-0.007 WP lift (softened) in classes 0/1, neutral in class 3 (where symreg already struggles). Total blend: GRU + 0.15*symreg_0 + 0.15*symreg_1 + 0.10*symreg_2 + 0.05*symreg_3.

**(d) Cheapest test:** Run scripts/symreg_explain.py per class on GRU residuals; extract top 1-2 terms per class; compute closed-form corrections; blend on valid with grid search. (~5 min, symreg already optimized).

**(e) Falsifier:** If symreg trees per class are nearly identical (same top terms), the class-conditional assumption is wrong. Classes are not structurally distinct enough to warrant separate explanations.

**(f) Research tree:** A4 (symbolic explanations, per class) → C2 (diverse members, MoE); branch: "class-partitioned symreg blend."

---

### 8. **Contrastive Validation Split: Adversarial Class Mixing**
**(a) Concept:** Re-split validation into two halves: (A) sequences with original class mix (40% 0/1, 34% 2, 26% 3); (B) sequences with *inverted* class mix (swap class 0↔3, 1↔2). Train a single GRU on A + B with data-augmentation flag. If model overfits to *specific class encodings* (e.g., class 3 = always use feature X), performance on B should collapse.  
**Source:** Adversarial validation (Blum et al.); domain adaptation; contrastive learning.

**(b) Analogy to data:** Test class mix is unknown (likely later period). If validation overstates test by ~0.02, it may be because valid has 40% classes 0/1 (easy) and test has fewer. By inverting class labels during validation, we measure *robustness to class mix shift*, not just accuracy.

**(c) Testable hypothesis:** Model trained with adversarial class-flip augmentation achieves softened-mask WP ~0.001 higher on inverted-class split than baseline (i.e., learns class-agnostic features). Gap to normal valid remains ~0.0015-0.002 (unavoidable).

**(d) Cheapest test:** Create inverted-class sequences (relabel class codes, recompute targets dynamically using class-conditional decoders in src/wnn/ticks.py); train GRU for 1 epoch on augmented set; measure valid + inverted_valid softened-mask WP. (~2 min, 1-epoch).

**(e) Falsifier:** If inverted-class WP is not meaningfully lower than normal valid WP, classes are already decorrelated in features. Class identity is not a confound.

**(f) Research tree:** A5 (shift across train/valid/test) → C1 (GRU recipe robustness); branch: "adversarial class-flip augmentation."

---

### 9. **Two-Horizon Readout: Separate t0/t1 Dynamics via Lagged GRU**
**(a) Concept:** Current GRU predicts both t0 and t1 from the same hidden state at time t. But t1 is typically easier (WP 0.672 vs 0.658 for t0). Hypothesis: t1 is "lagged" relative to t0 (reflects the state 1-2 rows ahead). Train a bifurcated GRU: one head reads h[t], another reads h[t-1] (or a memory buffer). Readout: t0 = W0 @ h[t], t1 = W1 @ h[t-1].  
**Source:** Temporal hierarchy in RNNs; Transformer-style positional encodings.

**(b) Analogy to data:** The target law is deterministic from book state, but t1 only requires knowing the spread (easier); t0 requires knowing the exact bid/ask level (harder). If t1 is naturally a *summary* of past state, it should be read from older history.

**(c) Testable hypothesis:** Bifurcated readout (t0 from h[t], t1 from h[t-1] or moving-average h) yields t0 WP ~0.665+ and maintains t1 WP ~0.672. Current single readout: t0 0.66, t1 0.67. Expected +0.005 on t0.

**(d) Cheapest test:** Add auxiliary output head in src/wnn/nn.py that reads from h[t-lag] instead of h[t]; train GRU for 1 epoch with separate t0 and t1 targets; measure softened-mask WP per target. (~2 min).

**(e) Falsifier:** If t0 WP doesn't improve (stays 0.66), t1 is not truly lagged; the targets are synchronized in time, and separate horizons are a misspecification.

**(f) Research tree:** C5 (aux-next: log rows to next target change) → A1 (target law asymmetry); branch: "lagged-horizon readout."

---

### 10. **Robust Ensemble Weighting via Minimax Regret (Class-Weighted)** *(unconventional)*
**(a) Concept:** Instead of averaging or blending models with fixed weights, use a game-theoretic weighting scheme: for each class, compute the "worst-case" blend weight that would minimize the maximum regret (loss diff from oracle). This is analogous to Kelly criterion in betting.  
**Source:** Minimax regret (Arora et al.); adversarial blend learning.

**(b) Analogy to data:** We have 4 classes with wildly different difficulties (class 3 WP 0.42-0.44, class 0 0.72+). A fixed global blend weight may over-weight easy classes. Minimax regret automatically down-weights members that fail on hard classes.

**(c) Testable hypothesis:** Minimax-regret class-weighted blend achieves softened-mask WP 0.6575+ (vs 0.6569 current blend), with class 3 WP +0.005 above baseline (due to adaptive down-weighting of weak members).

**(d) Cheapest test:** Collect per-class WP for each model (baseline, GRU); formulate regret matrix (4 classes x 2 models); solve minimax LP: `min_w max_k (w^T loss_k)`; evaluate blend on valid softened mask. (~1 min, LP solve).

**(e) Falsifier:** If minimax regret weights are similar to uniform (0.5, 0.5), the models already have balanced per-class performance. Regret is not an actionable signal.

**(f) Research tree:** C3 (ensemble weighting) → D1 (decision rule); branch: "minimax-regret adaptive blend."

---

### 11. **Feature Ranking via Class-Conditional Mutual Information Heat Map**
**(a) Concept:** For each class, compute MI(feature, target) conditioned on is_scored (filter out unscored rows). Rank features by MI. Then, hierarchically select top-K features per class (e.g., top 20 for class 3, top 15 for class 0); train a *separate lightweight model* (logistic or shallow net) for each class using only its top features.  
**Source:** Information-theoretic feature selection; conditional independence.

**(b) Analogy to data:** Our features are anonymized and transformed. Some features are "universal" (signal all classes), others are class-specific. MI reveals these dependencies without assumptions about linearity.

**(c) Testable hypothesis:** Class-conditional feature selection reduces model size (20-30% fewer params), improves test generalization (softened-mask WP +0.002), and yields interpretable feature sets per class.

**(d) Cheapest test:** Compute MI(feature_i, target) per class using sklearn.feature_selection or symreg; rank; train logistic regressions with top-K features per class; measure valid softened-mask WP per class. (~3 min, no NN training).

**(e) Falsifier:** If top-K features per class are highly overlapping (>80% overlap), feature selection isn't revealing class-specific structure. The features are already universal.

**(f) Research tree:** B1 (heuristic library, linear lift) → C2 (diverse members, lightweight); branch: "class-specific feature selection via MI."

---

### 12. **Prediction Variance Decomposition: Aleatoric vs Epistemic by Sequence**
**(a) Concept:** Train ensemble of K GRU seeds with *different initializations and dropout* on the same data. For each row, measure *epistemic uncertainty* = variance of predictions across seeds; measure *aleatoric uncertainty* = mean dropout-induced variance within each seed. High epistemic = model disagreement = risky. High aleatoric = inherent label noise. Weight predictions inversely by total uncertainty: `pred_final = sum(pred_k / (var_epistemic + var_aleatoric))`.  
**Source:** Bayesian deep learning (Kendall & Gal); uncertainty quantification.

**(b) Analogy to data:** Some rows are inherently noisy (e.g., during flash crashes, i1 influence dominates). Others are easy (rows far from price changes). Uncertainty weighting lets us *trust* predictions less on hard rows and more on easy rows.

**(c) Testable hypothesis:** Uncertainty-weighted ensemble (K=3 seeds, dropout_p=0.2) achieves softened-mask WP 0.658-0.660 (vs 0.6569 simple blend), with class 3 WP especially lifted (+0.003).

**(d) Cheapest test:** Train 3 GRU seeds (1 epoch each, cached) with different dropout; compute per-row epistemic and aleatoric variance; blend predictions with variance weighting; measure valid softened-mask WP. (~4 min, 3x1-epoch GRU).

**(e) Falsifier:** If uncertainty estimates are uncorrelated with actual error (corr < 0.3), the ensemble variance is not a proxy for prediction reliability. Uncertainty weighting is misdirected.

**(f) Research tree:** C3 (ensemble averaging) → D3 (recency weighting); branch: "uncertainty-weighted ensemble."

---

## Summary & Ranking by Feasibility + Upside

**High feasibility, high upside:**
- #5 (Blend orthogonalization): +0.003-0.005, 2 min
- #4 (Stale-row regime detection): +0.004-0.006, 4 min

**Medium feasibility, high upside:**
- #7 (Class-conditional symreg): +0.004-0.007, 5 min
- #12 (Uncertainty-weighted ensemble): +0.002-0.004, 4 min

**Unconventional / exploratory:**
- #3 (Cross-instrument label denoising): +0.005-0.010, 3 min ← *biggest potential*
- #6 (Adaptive softening lambda): +0.003-0.005 in correlation, 1 min
- #10 (Minimax regret blend): +0.001-0.006, 1 min ← *fastest*

**Foundational (may unlock larger gains):**
- #1 (Elliptical Wishart whitening): +0.002-0.004, 5 min
- #8 (Adversarial class mixing): +0.001-0.002, 2 min
- #2 (Tick-space PCA): exploratory, 2 min
- #9 (Lagged-horizon readout): +0.003-0.005, 2 min
