# Panel: brainteaser solver, the algebra of the metric and the target (2026-09-26)

Lens: identities and invariances of the pooled clipped |y|-weighted Pearson (WP) and of the target
law. Offline checks only: CPU at 6-8%, no training, no GPU. Data: all 1,873 valid sequences, 3,528,273
scored rows. Predictions are out of sample on all of valid: `ab30_eps_aux` (train-only GRU) and the
organizer baseline x0.25, blended 0.65/0.35 (the structure of the LB incumbent). Blend WP 0.6839.
"Cross-fitted" means fit on 3 chain folds and score the 4th (`decide._chains`). The ± is the
chain-bootstrap sd over 200 draws. Scripts are in the session scratchpad `alg/` (§6). The machine
was under load, so four planned checks were not run; the cards mark them as *untested*.

## 0. Verdict

There is no free lunch in the metric. Pointwise, |clip y|-weighted MSE targets exactly the
WP-optimal functional, and to second order it *is* WP up to a global affine map (§1). Fifteen causal
post-hoc maps each add <= +0.0007: class, nonlinear, cross-target, slow component, event interval,
event distance, member disagreement, jitter. All but one (interval, ≈0) lose in class 3 on held-out
folds (§3). The CE failure (E-072) was a scoring-rule mismatch, not a readout problem (§2). In class 3
the metric is P(stop-out) against P(win >= 2 ticks), and the 70% "inside" mass is worth ~nothing.

The levers the algebra leaves are all train-time: (A) Rao-Blackwellised class-3 labels (tower
property), (B) per-batch loss normalization, and (C) an under-dispersed sequence-level component.

## 1. WP is Pearson under a label-tilted measure (Q1)

Write ỹ = clip(y, ±2) and s = is_scored. Then WP_k = corr_Q(ỹ_k, p̃_k) with dQ_k ∝ |ỹ_k| s dP. Each
target has its own measure.

- **Optimum.** Over functions of the information set X, corr_Q is maximized exactly by
  {a + b p*, b > 0}, p* = E_Q[ỹ|X] = E[|ỹ|ỹ | X, s=1] / E[|ỹ| | X, s=1]. This is the projection
  theorem in L²(Q). More generally, the best post-hoc map of p̂ given causal side info z is
  E_Q[ỹ | p̂, z]: corr(y, E[y|G]) >= corr(y, f(G)) for any G-measurable f. That is what §3 fits.
  Since |p*| <= 2, the prediction clip never binds at the optimum. Measured: |p| > 2 on 1.2e-4
  (t0) and 2.0e-4 (t1) of scored rows.
- **Local form.** Take p = p* + ε and let ε⊥ be the residual of ε on span{1, p*}. Then
  ρ = ρ*·(1 − ½·Var_Q(ε⊥)/Var_Q(p*)) + O(ε³), while wMSE = Var_Q(η) + E_Q[ε²]. So wMSE is WP's exact
  second-order surrogate, except that it also charges global level and scale errors, which WP
  forgives. The functional is elicited by |ỹ|·Bregman_φ(ỹ, p) for any strictly convex φ. Their
  gradients are wMSE's gradient times φ''(p), a prediction-dependent row weight. The local form
  wants a flat weight, so φ = x², i.e. wMSE, is the right member.
- **Code check.** `weighted_mse_loss` clips y first and weights each target by its own |clip y|
  (src/wnn/nn.py:64), which matches the metric. Propensity weights are functions of x, so they
  leave p* unchanged and only move capacity. The final score is ½(ρ0 + ρ1), and each ρ is
  invariant to its own affine map, so the heads separate. The t0/t1 weighting in the loss only
  allocates capacity.
- **Clip regimes.** Share of Q-mass on rows with |y| > 2: class 0 18.5%, class 1 13-15%, class 2
  6.5-7%, class 3 3.5%. Nearly all of it is large winners. Rows whose loser constant −(sp+2)q
  exceeds 2: 1.7-2.2% per class (class 3: sp = 2, 1.7%). Because the loss clips before it weights,
  the optimum is the tilted mean of the *clipped* value in every regime. I found no regime where
  the wMSE optimum differs from the WP optimum.
- **Pooling.** The local loss is the pooled Q-MSE, with no class or segment weights. That explains
  the refutations of class-mass weights (E-104) and of per-class and per-segment blend weights
  (E-127, E-138). Those fits maximised per-segment WP, which rescales segments, and the pooled ρ
  charges for that.

## 2. The target is one stop-loss payoff of one price (Q2, Q4)

Work in ticks with D0 = F − ask, D1 = bid − F and spread sp. Then t_k = q·g_sp(D_k), where
g_sp(d) = d for d >= −sp and g_sp(d) = −(sp+2) otherwise. In words: a long and a short both exit at
F, and a trade that loses more than the spread is stopped at 2 extra ticks. Both targets are
functionals of one law, P(F − mid | x), with D0 + D1 = −sp known.

The metric sensitivity of the tilted mean T = Σπ|v|v / Σπ|v| to outcome j is
∂T/∂π_j = |v_j|(v_j − T) / Σπ|v|. Measured for class 3 with sp = 1 (98.3% of class-3 scored rows):

| t0 outcome (ticks) | stop −3 | F=bid −1 | F=ask 0 | +1 | +2 | >=+3 (clip 3.04) |
|---|---|---|---|---|---|---|
| share of rows | 0.143 | 0.327 | 0.375 | 0.094 | 0.037 | 0.026 |
| share of Q-mass | 0.430 | 0.328 | 0.000 | 0.094 | 0.074 | 0.074 |
| ∂T/∂π | −3.66 | +0.10 | 0 | +1.42 | +4.15 | +8.2..8.4 |

The t1 row is the mirror image: T = −0.816, stop −3.24, F=ask +0.15.

- **What the class-3 metric is.** A contest between P(stop-out) and P(win >= 2 ticks). The inside
  split (F at bid or at ask) holds 70% of rows but has sensitivity 0.10-0.15, and 37.5% of rows
  (t0 = 0) are invisible to t0. So how F's ties are rounded does not matter for scoring. The split is
  asymmetric (class 3: 53.5% F = ask, 64% of sequences above one half; class 2: F = bid 59% vs
  F = ask 33%), and that asymmetry is worth ~0.
- **Why CE lost (E-071/E-072).** Plain CE 0.6489 → |y|-tilted CE 0.6536 (+0.0047 weighting) →
  tilted CE trained *through* the readout with wMSE 0.6645 (+0.0109 scoring rule) → plain wMSE
  0.6651 (readout form −0.0006, parity). CE is proper for the whole 504-code distribution. Its Fisher geometry weights code j by 1/π_j,
  while WP weights it by |v_j|²(v_j − T)². So CE spends capacity on the high-entropy inside split
  and on winner sizes above the clip, and the metric ignores both. `outcomes.readout` already is
  the per-branch |y|-weighted expectation. The readout was right and the loss was wrong.
- **Branch head vs regression.** Both converge to the same p*. E-065's 0.968 means the per-branch
  values are almost constant given (class, sp). A 3-logit head with fixed value tables, trained
  through the ratio readout by wMSE, is therefore a reparameterisation. Offline proxy: a
  quadratic map of (p0, p1) per class, which sees whatever branch coordinates the two heads
  jointly encode, gains +0.0007 ± 0.0004 (§3). Expected gain of a branch head is ~0. Do not build it.

## 3. Pooling and causal post-hoc maps (Q3)

Cross-fitted Q-weighted least squares y_k ~ design, fitted per target. Δ is pooled WP against the
raw blend (0.6839). Folds are listed as f0/f1/f2/f3.

| design (per market class unless noted) | ΔWP ± sd | folds | within class 3 |
|---|---|---|---|
| global affine (invariance check) | 0.0000 | 0/0/0/0 | −0.0002 |
| affine a_c + b_c·p | +0.0003 ± 0.0002 | +2/+7/+9/−2 e-4 | −0.0007 |
| 7-knot spline in p | +0.0005 ± 0.0002 | +8/+9/+12/−3 | −0.0012 |
| (p_k, p_other) linear | +0.0003 ± 0.0002 | +2/+7/+10/−3 | −0.0008 |
| quadratic in (p0, p1) | +0.0007 ± 0.0004 | +14/+10/+10/−1 | −0.0014 (c0/c1 +0.0011/+0.0014) |
| member weights (GRU, base) per class | +0.0001 ± 0.0002 | −1/+6/+9/−6 | −0.0009 |
| member weights, global | −0.0001 ± 0.0001 | | 0.65/0.35 is already Q-optimal |
| + slow EMA(p) hl 500 / 5000 / running mean | +0.0003 / +0.0004 / +0.0001 | | −0.0011 / −0.0007 / −0.0017 |
| + running mean of p since last i0 event | +0.0006 ± 0.0003 | −1/+11/+15/−1 | +0.0001 (all classes ≈0) |
| x |GRU − base| gate / + slow EMA of it | +0.0004 / +0.0003 ± 0.0002 | | −0.0009 / −0.0008 |
| x prediction jitter EMA | +0.0004 ± 0.0002 | | −0.0011 |
| affine per rows-since-event bucket (10) | −0.0002 ± 0.0003 | −9/+1/+5/−2 | −0.0014 |
| affine per spread (from labels) | +0.0091 ± 0.0007 | leak: c0/c1 +0.019; causal c2/c3 +0.0005/−0.0007 | −0.0007 |
| E_Q[y | class, spread] alone | WP 0.362 | = E-065 class_sp 0.366 | level structure only |

Calibration is exhausted: every causal map is <= +0.0007 (1-3 sd), <= 0 on fold 3, and <= +0.0001 in
class 3. The only large row uses label-implied spread, which is not causal in classes 0/1 (E-074). It is
the same contamination as E-066/E-073. Where spread is causal (classes 2/3) it adds nothing. The
class-3 loss is systematic: the p → E_Q[y] map drifts across chains in class 3, so any refit of
class 3 on other folds hurts.

Variance anatomy under Q (t0; t1 within 0.03):

| component | share Var_Q(y) | share Var_Q(p) | share Cov | local corr | relative slope y~p |
|---|---|---|---|---|---|
| between class | 0.100 | 0.236 | 0.226 | 0.999 | 0.96 |
| between sequence, within class | 0.083 | 0.045 | 0.060 | 0.666 | **1.33** (t1 1.27) |
| within sequence | 0.817 | 0.718 | 0.714 | 0.633 | 0.99 |

Class levels are exact; raw Q-mean gaps (class 3 sp1: y −0.756 vs p −0.468) are the free global
affine map. The sequence-level component is under-dispersed: rescaling the non-causal sequence mean
of p by λ ≈ 1.35 would add ≈ +0.0018 (oracle); causal slow proxies recover +0.0001..+0.0004.

## 4. Other identities checked

- **Mirror t0 <-> t1** is exact in the law but not a column map (physicist F8). **Piecewise-constant
  targets** (E-001): the interval-mean smoother in §3 finds no exploitable within-interval jitter.
- **Per-batch loss normalization.** The loss is Σw(p−y)²/Σw per (32-sequence batch, chunk), so a row
  trains with weight w·W̄/W_b while the metric uses w. With a |y|·is_scored proxy weight, the CV of
  W_b is 0.20 at chunk 2000 and 0.28 at chunk 500. The 10-90% weight multipliers are 0.79-1.31 and
  0.73-1.48. The optimum does not change (W_b depends on y_i only at O(1/n)); only capacity moves.

## 5. Hypothesis cards (ranked by expected metric gain)

**A. H-RB3: Rao-Blackwellised class-3 labels (tower property on the exact F rule).** *Untested.*
Expected +0.001..+0.002, for one CPU label job and 2 fold arms.
- *Mechanism.* In class 3, y = q·g_sp(round(mean mid over the next 4 events) − quote). This is
  exact for 90-97% of rows (E-079, E-080). Condition on the first k future mid moves Z, and
  integrate moves k+1..4 with an x-free Markov-1 kernel K on mid increments. Train with weight
  m1 = E_K[|ỹ| | Z] and target m2/m1, where m2 = E_K[|ỹ|ỹ | Z]. If K is right, E[m1|x] = E[|ỹ||x]
  and E[m2|x] = E[|ỹ|ỹ|x], so p* does not move while the label noise from moves k+1..4 disappears.
  Class 3 has ~30 effective outcomes per sequence (physicist F3). The recipe is data-limited and
  memorises noise (E-152, E-153), so lower label variance acts like more data and more usable epochs.
- *Size.* Noise ratio ≈ (R²_k − R²_model)/(1 − R²_model), with class-3 R²_model ≈ 0.19 (WP 0.43).
  The first move plus spread explains R²_1 ≈ 0.60-0.67 (E-011 0.773; physicist F4 0.82). So the
  noise ratio for k = 1 is ≈ 0.5-0.6, roughly doubling effective class-3 data, which E-152 values
  at +0.002-0.003 for all classes. k = 3 is the safest choice (lowest bias) with a smaller gain.
- *Cheapest test.* First, CPU only: `scratchpad/alg/rb.py`, written but not run, computes R²_k for
  k = 0..4 on 3 class-3 row groups. Scale it to ~200 class-3 train sequences with
  `ticks.reconstruct` and `online_c3.rebuilt_labels`, and add the bias share
  Cov_Q(p̂, y − ỹ_k)/Cov_Q(p̂, y) for an out-of-sample GRU. Go if the noise ratio is <= 0.7 and the
  bias share is <= 0.05. Second: one fine-tune arm per fold (parent `ab29_mask_f{0,1}`, E-141
  recipe). Swap class-3 (target, weight) for (m2/m1, m1) only on rows where the rebuilt label with
  the true path equals the given label; leave all other rows unchanged.
- *Falsifier.* Bias share > 0.1, or the arm is <= parent at λ 0.02 on either fold.

**B. H-NORM: per-batch normalization pulls capacity away from the dense regime.** *Untested.*
Expected +0.0003..+0.0008, for a one-line change and 2 arms.
- *Mechanism.* The W̄/W_b multiplier (§4) under-weights densely scored, high-|y| chunks. Those
  carry 54% of the metric weight (E-139). Consistent with this, our levers help sparse sequences
  most (E-134, E-135), and chunk 500 raises the CV.
- *Test.* Divide by an EMA of W_b (or a constant fixed in epoch 0), fold-0/1 arms at 2 epochs.
  The mechanism predicts the gain concentrates in the dense tercile (E-132 slicing).
- *Falsifier.* Δ < 1 SE on both folds, or no dense-tercile concentration.

**C. H-SEQ: the valid fine-tune is sequence-level de-shrinkage; has it finished?** *Untested.*
Expected 0..+0.001, for one cached-prediction check.
- *Mechanism.* The between-sequence slope is 1.33 (§3). That matches the skeptic's split of the
  fine-tune gain: +0.0022 between vs +0.0008 within sequences on the holdout. Oracle headroom is
  +0.0018; causal slow proxies get <= +0.0004.
- *Cheapest test.* `scratchpad/alg/anat_ft.py` (written, not run) computes the anatomy slopes and
  the oracle rescale gain for `ab29_mask_f{f}` → `ab39_ftm_f{f}` → `ab45_lpft_f{f}` on their own
  held-out folds. If the slope stays >= 1.2 after the fine-tune, give the fine-tune more
  sequence-level signal: a longer schedule with wd 0.1 (E-153), or an aux head on the forward
  sequence-window Q-mean of y.
- *Falsifier.* Slope ≈ 1 after the fine-tune; then close this line.

**D. H-CAL02: quadratic (p0, p1) readout for classes 0-2 only.** *Untested on fine-tuned models.*
Expected +0.0003..+0.0008; possibly subsumed by LP-FT, which re-fits the head (E-148).
- *Mechanism.* A small nonlinearity of (p0, p1) → E_Q (+0.0011/+0.0014 within classes 0/1); class
  3 excluded because every class-3 map loses on held-out folds.
- *Test / falsifier.* Cross-fit the same design on `ab39_ftm_f{0,1}` / `ab45_lpft_f{0,1}` on their
  own folds (classes were chosen after §3, so only fresh models count); kill if <= 0 on either fold.

**E. Closures (do not spend).** CE/branch/outcome-code heads (the scoring rule failed; a wMSE ratio
readout is a reparameterisation). Tie rounding of F (sensitivity ≈0.1). All §3 post-hoc maps.

## 6. Reproduction

Session scratchpad `alg/`: `extract.py`/`extract2.py` (scored rows plus causal context), `m1.py`,
`m3.py`, `m4.py` (anatomy, sensitivity, tie asymmetry, batch-weight CV), `calib.py` (all maps,
chunked normal equations). Written, not run: `rounding.py`, `rb.py`, `anat_ft.py`, `m5.py`. Run with
`WNN_CPU_PCT=6 WNN_MEM_MAX=4G OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 scripts/limited`; at 6% of 16
CPUs `limited` exports 0 threads, so NUMBA_NUM_THREADS must be set.
