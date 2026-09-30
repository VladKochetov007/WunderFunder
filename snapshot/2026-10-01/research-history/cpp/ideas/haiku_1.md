# Haiku Idea Generator 1: Random Matrix Theory + FX Options

Generated from skim of RMT (Livan et al. 2017) and FX Options (Wystup 2006).

## Idea 1: Wigner Semicircle in Order-Book Feature Spectrum

**Book Concept:** Wigner's semicircle law (RMT Ch. 4-5, p. 24-40) describes how eigenvalues of Gaussian random matrices follow a semicircular spectral density. Eigenvalues scale with √N and concentrate in [-2√N, 2√N].

**Analogy:** Our 112 features form a covariance/correlation matrix. Compute eigenvalues of sliding-window feature covariance. Real market has structure; random noise has semicircle. Deviation from semicircle = information signal.

**Hypothesis:** Feature correlation matrices with eigenvalues far from semicircle law predict future price moves. Large eigenvalue outliers = dominant flow patterns; small eigenvalues = noise. WP improves with spectral deviation magnitude.

**Cheapest Test:** 
- Compute feature covariance on rolling windows (e.g., rows 1-99 → row 100)
- Eigendecompose, rescale by √N
- Fit theoretical semicircle ρ_SC(λ) = (1/π)√(4-λ²)
- Measure KL divergence or Kolmogorov-Smirnov distance to semicircle
- Add spectral distance as a scalar heuristic feature
- Test via symreg (src/wnn/symreg.py) to see if distance enters top terms

**Falsification:** If KS-distance to semicircle has zero correlation with (t0, t1) residuals in per-class analysis, idea fails.

**Tree:** A2 (heuristic library), B1 (features for GRU)

---

## Idea 2: Level Repulsion Strength as Bid-Ask Spread Predictor

**Book Concept:** Wigner's surmise (RMT Ch. 2, p. 11-12) shows eigenvalue spacings follow p(s) = (πs/2)exp(-πs²/4); spacings are repelled at small distances. Repulsion factor ∏|x_j - x_k|^β (Eq. 2.15) enforces anti-clustering.

**Analogy:** Order-book volume/depth at different levels repel each other due to market microstructure (no two limit orders at identical price). Compute spectral repulsion statistic on depth matrix; map to spread prediction.

**Hypothesis:** Stronger level repulsion (tighter eigenvalue spacing distribution) occurs in narrow-spread regimes (low volatility). Repulsion parameter β̂ (fitted to actual eigenvalue gaps) predicts next spread quantile.

**Cheapest Test:**
- Extract bid/ask depth matrix D from features (11 bid cols, presumably matching ask structure)
- Compute eigenvalues of D^T D (Wishart matrix)
- Fit empirical spacing distribution p(s) to observed gaps
- Estimate shape parameter from Kolmogorov-Smirnov test vs. Wigner's surmise
- Use repulsion metric as class-3 spread predictor
- Correlate with ground-truth spreads in cache/valid_targets.npz

**Falsification:** If repulsion metric ⊥ spread targets within class 3, fails.

**Tree:** A2 (depth decoding), B2 (class-3 specific feature)

---

## Idea 3: Marčenko-Pastur Excess as Correlation Leak Detector

**Book Concept:** Marčenko-Pastur (RMT Ch. 14, p. 84) gives limiting spectral density of Wishart X^T X / N when X is p × N with iid Gaussian entries; density depends on aspect ratio λ = p/N. Spectrum is compressed when λ → 0 (tall matrix).

**Analogy:** Our features are N_rows × 112. If actual eigenspectrum deviates from M-P prediction given our aspect ratio, it signals lookahead leak or information leakage from future (correlated to targets).

**Hypothesis:** Sequences with correlation spectrum far from M-P (especially bulging top eigenvalue) have artificially inflated correlation to targets; this inflates valid WP. Filtering sequences with large M-P excess reduces overfitting.

**Cheapest Test:**
- Per sequence: compute feature covariance, eigenvalues
- Compute theoretical M-P density for λ = 112 / sequence_length
- Measure maximum eigenvalue vs. M-P edge
- Flag sequences with top eigenvalue > M-P upper edge by >2σ
- Train GRU with/without flagged sequences; compare valid WP
- Check if removing sequences improves test robustness

**Falsification:** If flagged sequences have same test/valid gap as clean ones, no signal.

**Tree:** A3 (mask robustness), D1 (decision rule validation)

---

## Idea 4: Coulomb Particle Dynamics on Order-Book Depth Levels

**Book Concept:** Coulomb gas (RMT Ch. 4, p. 24-30) models eigenvalues as charged particles on a line under quadratic confinement + logarithmic repulsion. Energy V[x] = (1/2N)∑x_i² - (1/2N)∑ln|x_i - x_j|. Equilibrium density minimizes free energy.

**Analogy:** Bid-ask depth at price levels d₁, d₂, ... acts like particles; volume = charge. Depth concentrations are confined (liquidity clusters at support/resistance) and repel via microstructure (no crossing). Model depth profile as Coulomb gas equilibrium.

**Hypothesis:** Next price move direction follows "gradient" of depth particle distribution. If particles cluster above mid (bid volume > ask), price has upward momentum. Coulomb equilibrium predicts the drift.

**Cheapest Test:**
- Extract depth profile: bid_1, ..., bid_11, ask_1, ..., ask_11 from features
- Normalize depths as probability; treat as Coulomb particle positions
- Compute "center of mass" (weighted mean): COM = ∑d_i · vol_i / ∑vol_i
- Compare COM to mid-price; COM > mid → predict t0 > 0, t1 > 0
- Test via logistic regression on {COM > mid} vs. t0/t1 direction
- Use in heuristics (src/wnn/heuristics.py)

**Falsification:** If COM sign ⊥ target sign per class, fails.

**Tree:** B1 (heuristic library), A4 (symbolic explanation)

---

## Idea 5: Volatility Smile Across Market Classes (CRAZY)

**Book Concept:** Volatility smile (FX Options Ch. 1.3.3, p. 19-27) is non-flat implied volatility as function of strike/delta. Smile shape (curvature, skew) reveals risk-neutral distribution shape; risk reversals quantify skew, butterflies quantify kurtosis.

**Analogy:** Our four market classes have different target quantum (tick sizes). Within each class, the distribution of t0, t1 labels has different "smile": class 0/1 (small tick) = flatter smile; class 3 (large tick) = steeper smile due to quantization. Use smile geometry to predict.

**Hypothesis:** Fit smile surface to observed t0/t1 distribution per class. Smile curvature (volga analog) predicts target values better than raw Gaussian assumptions. Class-3 steep smile contains info about quantized outcomes.

**Cheapest Test:**
- Split validation set by class
- For each class, compute empirical distribution of t0 (and t1)
- Fit smile: σ_implied(K) = σ_ATM + α(K-F)² + β|K-F| (quadratic + linear skew)
- Extract smile parameters (smile_curvature, skew_slope) as class-specific features
- Add to heuristics; test correlation with prediction errors
- Use symreg to see if smile terms enter residual equations

**Falsification:** If smile parameters don't correlate with residuals, fails.

**Tree:** A1 (target law refinement), B1 (class-3 specific features)

---

## Idea 6: Greeks-Style Sensitivity Decomposition for Features (CRAZY)

**Book Concept:** Greeks (FX Options Ch. 1.2.3, p. 5-15) are derivatives of value w.r.t. parameters: delta (∂v/∂S), gamma (∂²v/∂S²), vega (∂v/∂σ), vanna (∂²v/∂σ∂S), volga (∂²v/∂σ²). Each Greek captures a distinct risk dimension.

**Analogy:** Instead of price sensitivity, compute "feature sensitivities" on t0: how much does moving feature i change the target? Use finite differences to compute:
- Delta_i = (t0(x+δe_i) - t0(x-δe_i)) / 2δ  [marginal effect]
- Gamma_i = (t0(x+δe_i) + t0(x-δe_i) - 2t0(x)) / δ²  [nonlinearity]
- Vanna_ij = ∂(∂t0/∂x_i) / ∂x_j  [interaction]

**Hypothesis:** Some features have high gamma (convex relationship to t0) or high vanna (strong interactions). These "explosive" features, when active, drive prediction confidence. Use Greeks as auxiliary loss terms in training.

**Cheapest Test:**
- On validation set, compute numerical Greeks for each sequence at rows 100-500
- Aggregate: mean delta, gamma, vanna per feature
- Features with high gamma should appear in symreg before low-gamma features
- Test GRU with auxiliary task: predict {gamma_i} alongside t0
- Check if auxiliary loss reduces t0 error (cross-task regularization)

**Falsification:** If auxiliary task doesn't improve WP or makes it worse, fails.

**Tree:** B1 (feature engineering), C2 (auxiliary heads)

---

## Idea 7: Barrier Knockout Probability for Scoring Mask

**Book Concept:** Barrier options (FX Options Ch. 1.5.1, p. 40-49) have knockout levels (barriers). "No-touch probability" (p. 219) is risk-neutral probability that price never hits barrier; computed via cumulative distribution under drift.

**Analogy:** Scoring rows (is_scored=1) occur when order-book undergoes certain events (price change, trigger rebalancing). Non-scored rows are "sheltered" from evaluation. Treat is_scored as a barrier-knock-in process: row becomes scored if certain levels (barrier levels in depth or volatility) are touched.

**Hypothesis:** Predict is_scored using no-touch probability logic. Rows with high distance to barriers (bid/ask extremes or volatility thresholds) stay non-scored longer. Use barrier-hit probability as propensity weighting: rows near barriers get higher weight in loss.

**Cheapest Test:**
- Extract bid, ask, depth range from features
- Define "barrier levels": max(bid), min(ask), peak volatility
- Compute distance of current row to nearest barrier
- Compute cumulative probability of hitting barrier in next k rows (assume Brownian motion locally)
- Use as weight w_barrier = 1 / P(not_hit_in_k)
- Train GRU with weights; compare to unweighted baseline
- Check if barrier-weighted loss improves valid WP

**Falsification:** If barrier weights hurt generalization, fails.

**Tree:** A3 (scoring mask process), D2 (weighting schemes)

---

## Idea 8: Spectral Persistence and Temporal Derivative of Eigenvalues

**Book Concept:** Random matrix theory typically studies static spectra. But eigenvalues of time-varying matrices evolve; their dynamics can be tracked. "Spectral edge" (RMT p. 875) is the boundary λ_max; when it shifts rapidly, structure changes.

**Analogy:** Our feature covariance matrix changes row-by-row. Track λ_max(t) over time. If λ_max grows rapidly, dominant correlation is forming (directional flow). If λ_max is stable, market is diffusive. Temporal derivative dλ_max/dt is a "volatility of volatility" signal.

**Hypothesis:** Price moves follow periods where spectral edges accelerate. dλ_max/dt > threshold → next 20 rows will have t0, t1 != 0. Use spectral acceleration as event detector.

**Cheapest Test:**
- Compute rolling window covariance eigenvalues at rows t, t-1, t-2
- Estimate dλ_max/dt ≈ (λ_max(t) - λ_max(t-1)) / 1
- Compute d²λ_max/dt² (second derivative = acceleration)
- Threshold: if d²λ_max > 0.1 (typical scale), flag next 20 rows as "high-move regime"
- Test: AUC on binary target (t0 != 0 in next 20 rows) given spectral acceleration
- Use as GRU auxiliary input (spectral_accel signal)

**Falsification:** If spectral acceleration AUC < 0.55, no signal.

**Tree:** B1 (features), C5 (aux heads for move prediction)

---

## Idea 9: Vanna-Volga Skew Explanation for Class-3 Bottleneck (CRAZY)

**Book Concept:** Vanna (∂²v/∂σ∂x, FX Options Eq. 1.20, p. 659) couples volatility and spot moves. Volga (∂²v/∂σ², Eq. 1.19, p. 654) is vol-of-vol. In exotic pricing, vanna-volga adjustment (p. 211-222) corrects Black-Scholes for smile curvature. Volga is large at out-of-money strikes (extreme quantiles).

**Analogy:** Class 3 (large tick) labels are quantized: {..., -1, -2, -3, 1, 2, 3}. Extreme quantiles (t0=-3, t0=3) are like OTM options: volga (curvature sensitivity) dominates. Small tick classes (0-2) have continuous-like targets, so volga is negligible. Class-3 model must account for volga.

**Hypothesis:** WP in class 3 improves by learning vanna-volga correction. Use asymmetric loss in class 3: larger loss for extreme quantiles if predicted smoothly across quantile boundaries. This forces GRU to learn discrete jumps.

**Cheapest Test:**
- Identify all rows with class 3
- Fit vanna-volga surface: loss correction = a · vanna_effect + b · volga_effect
- vanna ∝ skew in (t0, is_future_vol) cross-correlation
- volga ∝ convexity of loss w.r.t. prediction scale
- Train auxiliary head to predict {vanna, volga} terms on class 3
- Use as importance weighting: rows with high volga get 2× loss weight
- Compare valid WP (class 3 sub-metric) to baseline

**Falsification:** If class-3 WP doesn't improve, or gets worse, fails.

**Tree:** B2 (class-3 specific), C2 (model diversity)

---

## Idea 10: Free Probability Independence Between i0 and i1 (VERY CRAZY)

**Book Concept:** Free probability (RMT Ch. 17, p. 103-105) is a noncommutative generalization of probability. Two random variables are "free" if they are algebraically independent in a certain operator-algebra sense. Free addition: if A, B are free, then the spectral density of A+B is the free convolution ρ_A ⊞ ρ_B (not ordinary convolution).

**Analogy:** We have two instruments i0, i1. Are they independent or is there lead-lag? Treat feature subsets for i0 and i1 as free random variables. If they are free, their joint spectrum factors (approximately) as free product. If they are NOT free, spectrum shows anomalous peaks → lead-lag information.

**Hypothesis:** Deviation from free convolution assumption predicts cross-instrument price moves. Compute ρ_i0 (spectrum of i0 features) and ρ_i1 (spectrum of i1 features), compute free convolution ρ_i0 ⊞ ρ_i1, compare to observed joint spectrum. Distance = lead-lag signal.

**Cheapest Test:**
- Split 112 features into i0 (56 features) and i1 (56 features)
- Compute covariance eigenspectra for each, normalize
- Implement free addition (Voiculescu's R-transform method; approximation via simulation of free random matrices)
- Compute free convolution ρ_i0 ⊞ ρ_i1
- Compare KL divergence: observed spectrum vs. free convolution
- Large KS distance → flag row as having i1 lead-lag; add to heuristics
- Correlate with (t0, t1) copula structure (do t1 and i1_features move together?)

**Falsification:** If free convolution distance has zero correlation with t0/t1 comovement, fails.

**Tree:** B3 (cross-instrument), A2 (feature structure)

---

## Idea 11: Regime-Switching Stochastic Volatility for Quiet-Row Handling

**Book Concept:** Stochastic volatility (FX Options Ch. 1.3.11, p. 28-29) models σ_t as a hidden state that evolves randomly. In trading, regime-switching volatility (switching between high/low vol states) explains clustering and mean-reversion. Vol surface interpolation (p. 26-27) is the art of bridging market-quoted vols.

**Analogy:** B4 notes "stale-row regime" (>170 quiet rows): WP drops to 0.36 in this regime. Rows are partitioned into {trading_active, quiet_regime}. Quiet rows have low a0-a7 values (low intensity). Model quiet rows as low-vol regime state; use separate prediction head or hidden-state initialization for quiet periods.

**Hypothesis:** GRU with regime-switching hidden state: when entering quiet regime (detected by cumsum a0-a7), switch to low-variance emission distribution. This reduces spurious predictions. When exiting quiet regime, reset hidden state via volatility impulse.

**Cheapest Test:**
- Define regime as: quiet if sum(a0:a7, window_10) < percentile_30
- Train GRU_dual: one LSTM for active rows, one for quiet rows, with shared embedding but separate readout
- At each row, if regime changes, interpolate hidden state (vol-surface style: h_new = α h_active + (1-α) h_quiet)
- Compare validation WP to baseline GRU
- Check WP specifically in stale-row rows: expect improvement

**Falsification:** If quiet-row WP doesn't improve, fails.

**Tree:** B4 (stale-row regime), C1 (model recipe)

---

## Idea 12: Spectral Decomposition of Feature Interaction Terms (CRAZY)

**Book Concept:** RMT shows eigenvalue statistics reveal hidden structure. In quantum/statistical systems, "localization" vs. "delocalization" of eigenvectors determines physical properties. Localized eigenvectors concentrate mass on few coordinates; delocalized span the space.

**Analogy:** Feature interaction matrix M_ij = ⟨feature_i, feature_j⟩ has eigenvectors. Eigenvectors with high localization (entropy < threshold) correspond to specific feature pairs that strongly interact. Use localization of top k eigenvectors to identify key 2-way interactions.

**Hypothesis:** Top-k interacting feature pairs (identified by localized eigenvectors of M) have higher predictive power. Build shallow polynomial features from these pairs; add to GRU input. Model becomes hybrid: learned features (GRU) + deduced interactions (spectral).

**Cheapest Test:**
- Compute feature Gram matrix G = X^T X (normalized)
- Eigendecompose; compute IPR (inverse participation ratio) of eigenvectors: IPR_k = (∑v_i⁴) / (∑v_i²)²  [high IPR = localized]
- Select eigenvectors with IPR > median
- For each localized eigenvector, identify top-2 coordinate support (say coords i, j)
- Create feature: f_ij = x_i * x_j
- Add top-m such features to input; train GRU
- Compare WP to baseline

**Falsification:** If interaction features hurt generalization or don't improve WP, fails.

**Tree:** B1 (features), A4 (explanation)

---

## Summary Table

| # | Concept | Difficulty | Predicted Win | Branch |
|---|---------|-----------|---------------|--------|
| 1 | Wigner Semicircle Deviation | Low | Moderate | A2, B1 |
| 2 | Level Repulsion Spread | Low | Low | A2, B2 |
| 3 | Marčenko-Pastur Excess | Medium | High | A3, D1 |
| 4 | Coulomb Particle Depth | Low | Low-Mod | B1, A4 |
| 5 | Smile Across Classes | Medium | High | A1, B1 |
| 6 | Greeks Feature Sensitivity | Medium | Low-Mod | B1, C5 |
| 7 | Barrier Knockout Prob | Medium | Low | A3, D2 |
| 8 | Spectral Temporal Derivative | Low | Moderate | B1, C5 |
| 9 | Vanna-Volga Class-3 | High | High | B2, C2 |
| 10 | Free Probability i0/i1 | Very High | Very Low | B3, A2 |
| 11 | Regime-Switch Vol | Medium | High | B4, C1 |
| 12 | Spectral Eigenvector Interactions | Medium | Low-Mod | B1, A4 |

---

**Research Strategy:**
- **Quick wins** (1-2 hours): Ideas 1, 4, 8 as scalar heuristics (src/wnn/heuristics.py)
- **Medium-effort** (3-6 hours): Ideas 3, 5, 11 require training logic changes (model recipe)
- **Long shots** (>6 hours, high risk): Ideas 9, 10, 12 are experimental; use [[audit-before-goal-claim]] before reporting any gains

**Next Steps:**
1. Run symreg on ideas 1, 4 to see if spectral features enter top-100 terms
2. Train GRU variants for ideas 5, 11; compare valid WP class-wise
3. If any achieves >0.645 validation WP on class 3, audit and attempt test submission
