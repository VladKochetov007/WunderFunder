# Robustness under shift and the scoring-selection process (Opus shift scout, 2026-09-26)

Role: scout, ideator and specification critic for robustness to train/valid/test shift and for
how we select models. Hypotheses below were committed from the briefs, the ledger (E-060..E-105)
and my own checks before I read `research/tree.md` and `research/lit/*.md`. Verification tier B
for everything measured on validation and tier C for anything about the hidden test: only 4 LB
points exist, and the test mask is unobservable.

Tags: **[read]** = read in the source at the cited location; **[abs]** = verified from the
paper's abstract/landing page only; **[gk]** = general knowledge, not re-read; **[ours]** =
measured here. All measurements are tiny read-only checks: one process, `scripts/limited`,
`WNN_MEM_MAX=2G WNN_CPU_PCT=10`, no training, no GPU. They use only `cache/seqsums_*`,
`cache/valid_targets.npz`, `cache/pred_*_valid.npy` (mmap, one sequence at a time),
`cache/seqstats_{train,valid}.parquet` and `cache/chain_folds.npz`. Probe scripts are in the
session scratchpad (`scen.py`, `scen2.py`, `blends.py`, `dose.py`, `decomp.py`, `proxy.py`,
`effrob.py`, `calib.py`, `style.py`, `maskfrac.py`, `plainblend*.py`, `epsblend.py`). Each probe
below is described well enough to rerun.

## 0. Headline

1. **The validation → test ordering can be reproduced from validation alone.** Take the two thirds
   of validation sequences that look least like train (a feature-only adversarial score) and
   soften the mask to λ = 0.02-0.05. On that subset all three known LB orderings come out right,
   including the ab18 < baseline flip that no class-mix or λ variant reproduced before (validation
   scout §2.2). Full-valid λ=0 gets that flip wrong by 0.0044. See §2.
2. **Most of the rank-relevant shift goes through the scoring selection.** The rest is a generic
   difficulty increase that hits every model alike. The mask is an absolute-activity rule, so the
   scored fraction moves with the regime. The GRU's edge over the baseline halves on train-unlike
   sequences, but the between- and within-sequence parts degrade equally for all models. That
   argues against a GRU-specific shortcut. Class mix moves the *level* a lot and the *rankings*
   hardly at all.
3. **Consequences for current choices:** a larger propensity floor (ε 0.10 weakly dominates 0.03
   at every λ). Distillation from the baseline blend has the best effective robustness of any
   single model. Keep a baseline weight around 0.3 for eps+aux models. Predicted LB for
   eps_aux(+train/valid final fit) + 0.3·baseline: about 0.659, vs 0.6569 now (weak: 3 LB constraints).

## 1. New measurements [ours]

| id | probe | result |
|---|---|---|
| S1 | direction of train↔valid chain links (`cache/chains.npz`) | T→V 43 chains, V→T 37, T→V→T 5. No time order (agrees with both earlier scouts) |
| S2 | per-sequence scored fraction (rows ≥ 99), full valid | median 0.118 / 0.073 / 0.029 / 0.099 for classes 0-3; 5-95% ranges 0.050-0.220, 0.033-0.137, 0.010-0.067, 0.035-0.307; CV 0.41 / 0.40 / 0.55 / 0.69. Spearman with per-sequence `a3_mean` 0.27 / 0.40 / **0.92** / 0.67, with `a2_mean` 0.24 / 0.39 / 0.92 / 0.66. Scored rows come in short runs (median 2, mean 4.1, p99 34 rows). **The mask is an absolute-activity rule, not a per-sequence quota** |
| S3 | unscored \|y\| mass (60 seqs per class) | class 2 has 3.0% scored rows and 29× more \|y\| mass on unscored rows; at λ=0.02, unscored rows carry 37% of class-2 metric weight (classes 0/1/3: 10% / 17% / 11%). The softened metric is driven by class-2 off-mask rows |
| S4 | arm deltas vs ab20_e2 under λ ∈ {0,.02,.05,.1} × {valid, train} class mix | see table 1.1. Every mask-robust arm gains more as λ grows; class mix changes deltas by ≤ 0.0006 at λ ≤ 0.02 (≤ 0.001 at λ=.1) |
| S5 | blend with baseline×0.25 vs λ (full valid) | e2+0.5·base: +0.0017 / +0.0026 / +0.0036 / +0.0049 at λ 0/.02/.05/.1. eps_aux+0.3·base: +0.0011 / +0.0013 / +0.0016 / +0.0020 (0.5·base: +0.0004..+0.0017). **The baseline's blend value is mostly off-mask robustness; the ε floor already buys part of it** |
| S6 | adversarial terciles: within-class 5-fold HGB (100 iters, lr 0.1, 15 leaves) on the 902 feature-only per-sequence stats of `adversarial_validation.py` (2,000 train + 1,873 valid); OOF p(valid) per valid sequence; terciles within class | e2−base: +0.0101 / +0.0059 / +0.0055 (λ=0, near→far), +0.0080 / +0.0033 / +0.0033 (λ=.05). ab18−base: +0.0058 / +0.0017 / +0.0015 and +0.0023 / **−0.0019 / −0.0016**. e2 level 0.709 / 0.668 / 0.664. kNN-distance terciles give no monotone pattern; the classifier score is the useful shift coordinate |
| S7 | "stress proxy" = far 2/3 (terciles 1-2) with chain bootstrap (300 draws) | see §2 |
| S8 | pooled WP split into between-sequence (level) and within-sequence correlation, by tercile | e2: level corr 0.894 → 0.849, within corr 0.661 → 0.612; baseline 0.877 → 0.844 and 0.652 → 0.606. Same degradation for both. Predictions put 28-34% of variance between sequences vs 18-19% for targets, but the optimal between/within balance α* = (cov_b/vp_b)/(cov_w/vp_w) is 1.03-1.04: **no level-shrinkage gain** |
| S9 | effective robustness: fit far = a·near + b over 20 cached single models, report residuals | λ=0: slope 0.72; λ=.05: slope 0.79. Residuals (λ 0 / .05): baseline +0.0010 / +0.0020, **ab30_eps_aux_distill +0.0015 / +0.0014**, ab31_classw_eps +0.0007 / +0.0008, eps_aux +0.0005 / +0.0005, eps010 +0.0001 / +0.0002, eps003 −0.0004 / −0.0002, e2 −0.0005 / −0.0009, **ab18 −0.0015 / −0.0014**, gbm −0.0086 / −0.0064 |
| S10 | per-class output scale / affine recalibration, cross-fit chain folds {0,1}↔{2,3} (exact on sums, ±2 clip ignored) | GRUs: scale-only ≤ +0.0001; affine ≤ +0.0005 (λ=.05), fitted class scales 0.94-1.01. **Our GRUs are already class-calibrated.** baseline×0.25: +0.0013..+0.0028, class scales ≈ 1 / 1.04 / 1.13 / 1.21 (class 3 under-scaled) |
| S11 | best baseline blend weight per tercile (w ∈ {.3,.5,.7}) | e2: 0.3 / 0.5 / 0.5 (λ=0), gap ≤ 0.0005; eps_aux: 0.3 in every tercile and λ. Sequence-level oracle gating is worth ≤ +0.0005 |
| S12 | which inputs carry the regime "style" (median single-feature adversarial AUC of per-sequence means, classes 0-3) | i0 volume codes 0.75 / 0.65 / 0.69 / 0.61; i1 volume 0.57 / 0.80 / 0.69 / 0.63; i0 price codes 0.56 / 0.73 / 0.76 / 0.61; i1 price 0.52-0.59; a-cols 0.58-0.70. Within-class Spearman of per-sequence volume-code means with per-sequence target level is weak and **unstable** between train and valid (i0_v0 class 0: 0.35 vs 0.13; i0_v3 class 1: 0.09 vs 0.22; i1_v21 class 2: −0.13 vs +0.10). The a-column levels are **stable** (a6 class 0: 0.46 vs 0.46; a0: 0.38 vs 0.36) |
| S13 | label shift per class (seqstats, all rows) | class-3 within-sequence t0 std 0.885 (train) vs 0.691 (valid); zero-target share 0.293 vs 0.344. Between-sequence std of sequence-mean t0 is larger in train in every class (0.157/0.113/0.177/0.162 vs 0.101/0.096/0.123/0.111): train covers more heterogeneous regimes |
| S14 | cache integrity | `cache/pred_ab25_onlinenorm_valid.npy` is all zeros (`predict_run.py` has no online-norm path). The −0.68 "delta" in `artifacts/logs/q1.out` is this artifact. **The online-norm arm has only ever been scored on devB during training** (0.6702 vs 0.6707) |
| S15 | diversity along the selection axis | 0.5·eps003 + 0.5·eps010 (same seed) over the best single: +0.0004 / +0.0003 (far, λ 0/.05). Blending the old uniform-weight GRU (ab12, 0.619) into eps_aux: ≤ +0.0005 at w=0.1, negative at w ≥ 0.2. ab12 is worse than eps_aux even on *all* rows (λ=1: 0.466 vs 0.510) |

### 1.1 Arm deltas vs ab20_gru128_e2 (full valid; valid mix | train mix) [ours]

| arm | λ=0 | .02 | .05 | .10 | train mix λ=0 | .02 | .05 | .10 |
|---|---|---|---|---|---|---|---|---|
| baseline×0.25 | −.0072 | −.0062 | −.0049 | −.0034 | −.0071 | −.0060 | −.0046 | −.0030 |
| ab18 (p², no floor) | −.0042 | −.0047 | −.0053 | −.0060 | −.0049 | −.0055 | −.0062 | −.0070 |
| eps 0.01 | +.0012 | +.0018 | +.0025 | +.0033 | +.0014 | +.0021 | +.0029 | +.0039 |
| eps 0.03 | +.0020 | +.0029 | +.0039 | +.0052 | +.0022 | +.0033 | +.0046 | +.0060 |
| **eps 0.10** | +.0020 | +.0031 | +.0044 | +.0060 | +.0024 | +.0037 | +.0052 | +.0070 |
| eps .03 + aux-next | +.0030 | +.0038 | +.0048 | +.0061 | +.0033 | +.0044 | +.0056 | +.0071 |
| distill3 | +.0013 | +.0015 | +.0018 | +.0021 | +.0015 | +.0017 | +.0020 | +.0023 |
| train+valid OOF | +.0014 | +.0015 | +.0016 | +.0017 | +.0012 | +.0013 | +.0013 | +.0014 |

ε 0.10 weakly dominates ε 0.03 in all 8 cells. It "failed" in E-102 only because the rule looked at
λ ≤ 0.02, where the gap is 0.0000-0.0002.

## 2. The LB-calibrated stress proxy (selection result)

Three LB orderings are known (E-077, E-086): baseline×0.25 − ab18 = **+0.0015**; final_tv_e2 −
baseline = +0.0045, i.e. about **+0.003** for the train-only e2 after removing the +0.0014
train+valid gain (E-097); blend(base, final_tv) − final_tv = **+0.0021**.

| pair | full valid λ=0 | far-2/3 λ=0 | far-2/3 λ=.05 (chain-boot SD; 4 chain folds) | LB |
|---|---|---|---|---|
| base − ab18 | −0.0029 | −0.0017 ± 0.0013 | **+0.0017 ± 0.0011** (+.0016 +.0018 −.0003 +.0034) | +0.0015 |
| e2 − base | +0.0072 | +0.0058 ± 0.0010 | **+0.0034 ± 0.0010** (+.0032 +.0028 +.0051 +.0024) | ≈ +0.003 |
| 0.5 blend − e2 | +0.0018 | **+0.0020 ± 0.0005** | +0.0039 ± 0.0004 | +0.0021 |

So far-2/3 with λ between 0.02 and 0.05 matches all three orderings. The level is also consistent:
e2 scores 0.6661 at far λ=0 and 0.6473 at λ=.02, against an estimated train-only LB of about 0.653.
Caveats: this is 3 constraints fit with 2 knobs (subset, λ), and I picked the terciles after
seeing the monotone pattern. Public-LB noise is unknown. The adversarial score uses no targets
and no mask. Treat the proxy as the best-supported *stress cell*, not as an objective to tune
against.

Scores of cached candidates on the proxy (far 2/3, λ=.02 / .05): e2 0.6473 / 0.6249;
eps003 0.6500 / 0.6288; eps010 0.6504 / 0.6294; eps_aux 0.6512 / 0.6299; eps_aux_distill 0.6518 /
0.6305; e2+0.5·base (≈ the current LB blend) 0.6501 / 0.6289; **eps_aux+0.3·base 0.6525 / 0.6316**.
Predicted LB, if the train+valid final fit shifts both alike: eps_aux+0.3·base ≈ 0.6569 + 0.0024 ≈
0.659; the single distilled model ≈ 0.6586 at half the latency.

## 3. Diagnosis: where shift enters the pipeline

**Features (covariate shift).** Train and valid are blocks of different streams (S1; 0.99 AUC,
E-094). The "style" of a regime lives in per-sequence levels of the volume codes (i0 in classes
0/2, i1 in class 1), in activity statistics (a2/a3 std, a5-a7), and in book-shape price codes in
classes 1/2 (S12). Normalization is one global mean/std fit on 64 train sequences
(`fit_normalization`, 2 batches, every 7th row; the sentinel −5.19934 inflates the std), so an
unseen regime lands in off-support input regions. The GRU's advantage over the baseline halves on
train-unlike sequences (S6). The decomposition (S8) shows that all models lose level accuracy
(0.89 → 0.85) and within-sequence accuracy (0.66 → 0.61) alike there. That is a harder regime,
not a GRU-specific level shortcut. The volume-level ↔ target-level relation is weak and flips
sign across splits (S12), so any use of it is a regime crutch with no stable payoff. a-column
levels are the stable exception.

**Labels (conditional/label shift).** The target law and quanta are structural and stable. The
volatility regime is not: class-3 sequences are 28% more volatile in train than in valid, and
train is more heterogeneous in every class (S13). With |y|-weighted MSE, the volatile train regimes
dominate the gradient. On the metric side the per-class calibration still transfers (S10).

**Mask (selection shift), the channel that moves rankings.** The scored fraction is set by
absolute activity (S2: ρ = 0.92 with a3 in class 2). When the regime changes, the mask changes.
The propensity net is fit on valid regimes (devA) and then weights *train* regimes that are 0.99
separable from them, so the training weights are extrapolations (E-099: class 3 over-predicted
1.42×). On the test, the mismatch between our weights and the true scored set behaves like a
softer mask. The class-2 off-mask rows (S3) dominate that softness, and class 2 is exactly where
p² starves the loss (E-099: 4% vs 8% of weight). Every arm that spreads weight off the propensity
peak gains more as λ grows (table 1.1, S5, S9).

**Class mix.** It moves levels a lot (baseline 0.6728 → 0.6273 at the train mix) and rankings
hardly at all (table 1.1). It is irrelevant for selection and relevant for level forecasts only
(consistent with E-104).

**Per-instrument scale.** Our GRUs are calibrated per class (S10). The baseline×0.25 is
under-scaled in class 3 by about 1.2×, which matters only inside a blend (M4).

## 4. Ranked methods

Ranking is by expected LB gain per unit cost under the stress proxy. ★ marks an unconventional
idea (5 of 10). Common test harness: `scripts/train_rnn.py` with the ab30 recipe (`--propensity
prop_v1 --prop-power 2 --loss wmse --aux-struct 0.3 --hidden 128 --epochs 2 --seed 3`, about 15 min),
`predict_run.py`, then `decide.py compare` extended with the stress cells of M1.

### M1 ★ Stress-panel selection rule with an effective-robustness readout (no training)
- **Sources.** Petrik, *Robust RL* tutorial (`RobustRL.pdf`): slide 54, "the right size of the
  ambiguity set … depends on why there are errors"; slides 64-65, the percentile criterion
  max y s.t. P[return ≥ y] ≥ 1−δ; slides 83-84, robust *baseline regret* max_π min_ξ
  [return(π,ξ) − return(π_B,ξ)], whose regret bound is never worse than the baseline's [read].
  Charles & Darné 2009, *Variance ratio tests…* (`Heteroscedasticity.pdf`) §3.1 eq. 16-17:
  Chow-Denning joint test = max over m statistics with the Šidák bound α* = 1−(1−α)^{1/m} [read].
  Miller et al. 2021 (arXiv:2107.04649) ID/OOD accuracy on a line; Taori et al. 2020
  (arXiv:2007.00644) effective robustness [abs].
- **Mechanism.** The ambiguity set is our actual uncertainty: mask softness λ, regime novelty and
  chain luck. We decide on the worst plausible cell against the incumbent (baseline regret), not
  on the nominal cell. S7 shows the nominal cell (full valid, λ=0) mis-ranks an LB pair that the
  stress cell gets right.
- **Exact change.** (1) Extend `scripts/adversarial_validation.py` to save OOF p(valid) per valid
  sequence (within class, 5-fold HGB, as in S6) to a small `cache/advscore_valid.npz`. (2) In
  `decide.py` add `--stress`: cells = {full, far-2/3} × λ {0, .02, .05, .10} × 4 chain folds.
  Primary = far-2/3 at λ .02 and .05. Adopt B only if min over the 2 primary cells of Δ/SE >
  2.24 (Šidák, m=2, α=.05), no chain-fold sign flip in them, and no cell with Δ < −2 SE. (3) Print
  the effective-robustness residual far − (0.79·near + 0.114) at λ=.05 against the S9 line.
- **Cheapest discriminating test.** Done retrospectively (S7). Forward: the next LB pair on which
  plain and stress disagree.
- **Predicted effect.** No score by itself. It flips three current calls: ε 0.10 over 0.03
  (far .05: +0.0006 ± 0.0003, 3/4 folds); distilled eps_aux over plain eps_aux (+0.0006 far .05);
  baseline weight 0.5 for e2-type and 0.3 for eps+aux models.
- **Falsifier.** A future LB pair where the stress cell and full valid disagree and full valid is
  right. Or the far-2/3 subset stops reproducing base > ab18 once the adversarial model is refit
  with all 10.6k train sequences.

### M2 Mask-hedged weights: larger, support-aware or randomized propensity floor
- **Sources.** Shimodaira 2000, JSPI 90:227: flattened weights w^λ help under misspecification
  [gk; abstract per the independent scout]. Yamada et al. 2013, *Neural Computation* 25(5)
  (arXiv:1106.4729), relative ratio r_α = p/(αp + (1−α)q), bounded by 1/α [abs]. Byrd & Lipton 2019
  (arXiv:1812.03372): importance weights matter early in training and fade later, so our 2-epoch
  regime is where weights bite [abs]. Laroche et al. ICML 2019 (arXiv:1712.06924), SPIBB: fall
  back to the baseline where data are scarce [abs]. Petrik slides 64-65 for the randomized variant [read].
- **Mechanism.** Because the loss divides by Σw, p^a + ε ≡ (1−λ)p^a + λ with λ = ε/(1+ε). The
  floor *is* a relative-density weight. For a test mask that is λ-softer than validation, the
  Bayes-consistent row weight is E[m + λ(1−m) | x] = (1−λ)p + λ. The floor should therefore
  track the plausible λ range (0.02-0.05 against true scores, higher against our miscalibrated p̂),
  not λ=0. Train sequences whose regime is unlike any valid regime get extrapolated p̂, and they
  deserve a larger floor there.
- **Exact change.** (a) `--prop-eps 0.1` and `0.2` together with `--aux-next 0.3` (ε 0.10 was only
  run without aux). (b) Support-aware floor: per train `row_group`, s_i = OOF P(valid-like)
  from a seqstats adversarial model (valid sequences s=1). Return s_i from `Sequences.__getitem__`
  and set `main_valid = valid * (p**a + eps + kappa * (1 - s_i))` with κ = 0.1. (c) Randomized floor:
  per batch, ε = exp(U[log .01, log .3]).
- **Cheapest discriminating test.** Arms (a) 0.1 and (c) first (2 × 15 min), vs ab30_eps_aux on the
  M1 stress cells. Arm (b) needs per-sequence stats for all 10.6k train sequences (request R4).
- **Predicted effect.** Full valid λ=0 within ±0.0002. Far λ .05 +0.0005..+0.001. LB +0.0005..+0.001.
- **Falsifier.** ε 0.1+aux ≤ ε 0.03+aux in both primary stress cells, or (c) below (a).

### M3 Distil the stress-optimal blend into the deployed GRU
- **Sources.** Hinton et al. 2015 (arXiv:1503.02531) [gk]; Menon et al. 2020 (arXiv:2005.10419)
  §3, lower-variance distilled risk (per the independent scout [read]); Wortsman et al. CVPR 2022
  (arXiv:2109.01903): robustness of a robust endpoint carries over at no inference cost [abs].
  Our S9: `ab30_eps_aux_distill` has the highest single-model effective robustness (+0.0015 /
  +0.0014).
- **Mechanism.** The baseline's contribution is off-mask robustness (S5). A student trained toward
  the blend inherits it on low-propensity rows at single-model latency (about 41 vs 94 us/row). That
  frees budget for a 2-member blend of robust students, or student + baseline.
- **Exact change.** Tilt the teacher toward the stress-optimal weight (baseline 0.5 for e2-type
  members) while keeping unit scale: `--prop-eps 0.1 --aux-next 0.3 --distill
  "ab20_gru128_e2*0.75,ab23_e2_seed4*0.75,baseline*0.375" --distill-alpha 0.5`. The code averages
  teachers, so this gives 0.25·e2 + 0.25·seed4 + 0.5·(base×0.25). Arm 2: `--distill-alpha 0.7`.
- **Cheapest discriminating test.** One run (15 min) vs `ab30_eps_aux_distill` on the stress cells,
  then its 0.3-baseline blend vs eps_aux+0.3·base.
- **Predicted effect.** Far λ .05 +0.0003..+0.0008 over ab30_eps_aux_distill; LB about +0.0005 at
  the same latency.
- **Falsifier.** Student ≤ ab30_eps_aux_distill in both primary cells, or its effective-robustness
  residual ≤ +0.0010.

### M4 ★ Blend weights as a robust index-tracking portfolio (CPU only)
- **Sources.** Benidis, Feng & Palomar 2018, *Optimization Methods for Financial Index Tracking*,
  FnT Optimization 3(3) (`Palomar_FinEng.pdf`): §3.3 eq. 3.21-3.25, tracking error, downside risk
  DR = ‖(r_b − Xw)_+‖², Huber-robust HETE/HDR; §3.4 eq. 3.28, long-only budget w ≥ 0, 1ᵀw = 1;
  §3.5 eq. 3.33, cardinality/holding constraints; §5.4, robust measures stay on target in extreme
  periods [read]. Roadmap Ultimate Edition ch. 13 p.35, "Regime-based ML … portfolio weights
  conditioned on the regime" [read]. DeMiguel et al. 2009 on 1/N [gk].
- **Mechanism.** The blend is a portfolio of models tracking y. WP of a blend is a Sharpe-like
  ratio whose maximizer is the tangency-type OLS stack, which overfits. Palomar's constraints
  (long-only, budget, cardinality ≤ 3 for latency) plus a *downside* objective (penalize only
  cells where the blend trails the incumbent, Huber for robustness) give a stress-robust
  allocation. S10 adds a free member transform: baseline rescaled per class (1, 1.04, 1.13, 1.2).
- **Exact change.** New CPU script. One pass over K ≤ 6 cached predictions stores, per sequence ×
  target × {scored, unscored}: Σw, Σwy, Σwy², Σw·p_k, Σw·y·p_k and the K×K matrix Σw·p_j·p_k
  (the "Gram cache", 36 numbers). Blend WP for any β, λ, sequence weights or per-class member
  scale then follows exactly (the ±2 clip is ignored; verify the winner with `decide.py blend`).
  Maximize the mean over the M1 cells minus the Huber downside vs the incumbent, subject to β ≥ 0,
  Σβ = 1, ≤ 3 members. Cross-fit on chain folds {0,1}↔{2,3}.
- **Cheapest discriminating test.** CPU, about 10 min, ≤ 3 GB, members {eps_aux, eps_aux_distill,
  eps010, seed4, base×0.25, base×0.25 per-class-scaled}. Compare with 1/N and the fixed 0.7/0.3.
- **Predicted effect.** +0.0003..+0.0008 in the stress cells over eps_aux+0.3·base.
- **Falsifier.** Cross-fitted weights ≤ 1/N or ≤ the fixed 0.3 blend in the primary cells.

### M5 ★ MixStyle / domain randomization of the volume-code "style"
- **Sources.** Zhou et al. ICLR 2021 (arXiv:2104.02008), MixStyle: mix instance-level feature
  statistics across domains at train time to synthesize new domains [abs]. Tobin et al. IROS 2017
  (arXiv:1703.06907), domain randomization: "the real world may appear to the model as just
  another variation" [abs]. Evidence: S12 (volume-code means are the style carriers, their link
  to target level is unstable) and S6/S9 (the gap to close lives on train-unlike sequences).
- **Mechanism.** Per-sequence volume-code levels identify the regime but do not predict y stably.
  Randomizing them during training removes the incentive to key on them, while within-sequence
  deviations (the content the symreg terms use, E-100/E-101: v0 − EMA32(v0), z_v0) are kept.
  Price codes (spread/level information) and a-columns (stable level relations) are left alone.
- **Exact change** (train only, zero inference cost). In the training loop, before
  `build_inputs`, for the 44 volume columns (i0 22-43, i1 74-95): with probability 0.5 per
  sequence pick a partner j in the batch with the same `sequence_class(y)`. Draw λ ~ Beta(0.3, 0.3)
  and set x_V ← σ_mix·(x_V − μ_i)/σ_i + μ_mix, where μ, σ are the per-sequence column mean/std over
  non-sentinel rows and μ_mix = λμ_i + (1−λ)μ_j (same for σ). Leave sentinel values untouched.
  Event flags survive because an affine map preserves (in)equality. Variant DR: per-sequence
  offsets N(0, 0.25²) per side-block plus scale LogN(0, 0.1).
- **Cheapest discriminating test.** One train-only run with the ab30 recipe + MixStyle; read the
  M1 cells and the S9 residual.
- **Predicted effect.** Full λ=0 −0.001..+0.0005; far +0..+0.002; effective robustness
  +0.0005..+0.002; LB +0..+0.0015.
- **Falsifier.** S9 residual ≤ eps_aux's (+0.0005), or far λ=.05 Δ < 0.

### M6 ★ Causal de-leveling of the volume block (replace, not append), incl. Riemannian re-centering
- **Sources.** Avellaneda-Lee s-scores as implemented in Vertox, *Stat. Arb. … in Crypto, Part 2*
  (`Naive/…Part 2.pdf`): p.2-6, regress on eigen-portfolio factors and keep residuals; p.8,
  normalize the residual and trade the z-score; p.10, the author's own warning that fitting
  and trading on the same data is look-ahead [read]. Charles & Darné §2.3 eq. 10-11: Wright's rank
  scores r2,t = Φ⁻¹(r(x_t)/(T+1)); sign tests are exact under conditional heteroscedasticity [read].
  Zanini et al. 2018, IEEE TBME (PubMed 28841546): affine re-centering of each session's covariances
  to a reference makes sessions comparable (up to +30% cross-subject accuracy) [abs]. Ayadi,
  Bouchard & Pascal, ICASSP 2023 (`RobustEstimation.pdf`) §4: robust t-/Kotz-Wishart MLE of the
  center of SPD matrices [read]. Kim et al. RevIN, ICLR 2022 [gk].
- **Mechanism.** Same target as M5, enforced at inference: the network only sees volume codes
  relative to the sequence's own causal equilibrium. ab25 *appended* cumulative z-scores next to
  raw levels, so the crutch stayed available, and it was never scored beyond devB (S14).
- **Exact change.** First fix `predict_run.py` to honor `--online-norm`, re-predict ab25 and score
  it on the stress panel (about 3 min GPU). Then variant S: replace the 44 volume inputs by EWMA
  s-scores (x − m_t)/sqrt(v_t + 1e-3), half-life 2,000 rows, warm-started from the train global
  mean/var, plus one clipped raw side-level channel per block. Variant R: x̃ = G^{1/2} C_t^{−1/2}
  (x − m_t) per 11-dim side block, where C_t is the EWMA covariance refreshed every 100 rows and G
  is the train t-Wishart center (Ayadi et al.). Inference cost: S about 0.2 us/row, R about 2 us/row
  amortized.
- **Cheapest discriminating test.** The ab25 re-score (free), then one run of variant S.
- **Predicted effect.** Full λ=0 −0.002..+0.001; S9 residual positive if volume levels are
  crutches; LB −0.001..+0.0015.
- **Falsifier.** Far λ=.05 Δ < 0 and S9 residual ≤ 0.

### M7 Group DRO / CVaR over regime clusters with reference-normalized loss
- **Sources.** Sagawa et al. ICLR 2020 (arXiv:1911.08731): group DRO needs strong ℓ2 or early
  stopping to help worst groups [abs]. Levy et al. NeurIPS 2020, minibatch CVaR/χ² DRO [gk].
  Petrik slide 19: max-min over an ambiguity set; slide 54: its size should come from the error
  source [read].
- **Mechanism.** Up-weight regime clusters where the model is worse than a reference, so shared
  capacity is not spent on dominant train regimes. The loss is normalized by the reference because
  raw losses would just re-weight toward class 3 (inherently harder).
- **Exact change.** Groups = k-means (k=8 per class) on standardized seqstats of train sequences.
  Reference per-sequence loss from `cache/pred_ab20_gru128_e2_train.npy` (exists) on the same rows.
  Group weights q_g ← q_g·exp(0.01·mean(L_i/L_i^ref − 1)), normalized; main loss Σ q_{g(i)} L_i;
  weight decay 3e-2.
- **Cheapest discriminating test.** One run; M1 cells.
- **Predicted effect.** Full −0.001..0; far +0..+0.001. Evidence against it: S8 shows the far-regime
  loss is shared by all models, while DRO targets worst groups and our metric averages.
- **Falsifier.** Far λ=.05 Δ ≤ 0.

### M8 Volatility-regime normalization of the loss (heteroscedasticity)
- **Sources.** Charles & Darné §2.1 (Lo-MacKinlay heteroskedasticity-robust M2(k)) and §2.3 (sign
  tests exact under conditional heteroscedasticity) [read]; GLS [gk]. Evidence: S13 (class-3
  volatility 0.885 vs 0.691).
- **Mechanism.** |y|-weighted MSE lets volatile train regimes dominate the gradient. Normalizing
  per-sequence influence by its volatility equalizes regimes whose volatility differs by split.
- **Exact change.** Training-only multiplier r_i = (σ̄_class/σ_i)^0.5 on `main_valid`, with σ_i the
  |y|-weighted target std of sequence i (label-derived, training-only, never an input).
- **Cheapest discriminating test.** One run.
- **Predicted effect.** ±0.001.
- **Falsifier.** Far λ=.05 Δ ≤ 0, or class-3 far WP drops by > 0.002.

### M9 WiSE-style weight interpolation along the selection axis (downgraded)
- **Source.** Wortsman et al. CVPR 2022 (arXiv:2109.01903): interpolating weights between a robust
  and a specialized model improves OOD accuracy at no inference cost [abs].
- **Why downgraded.** The output-space proxies (S15) are small: blending two floors +0.0004, and
  the old uniform-weight GRU ≤ +0.0005. A useful robust endpoint would have to come from M5/M6.
  Only then: parent = M5/M6 model, child = 0.5-epoch fine-tune with p²+ε weights, sweep
  α ∈ {0, .25, .5, .75, 1} on the stress cells.
- **Falsifier.** Best α ∈ {0, 1}.

### M10 ★ Novelty-gated baseline bootstrapping (SPIBB; downgraded)
- **Source.** Laroche et al. ICML 2019 (arXiv:1712.06924): use the baseline where uncertainty is
  high [abs]. Petrik slide 83 [read].
- **Why downgraded.** The sequence-level oracle version (the best blend weight per adversarial
  tercile) is worth ≤ +0.0005 (S11). A causal row-level gate (running Mahalanobis distance of
  volume/a-column running means to the train class centroid) can only do worse. Revisit only if
  M5/M6 widen the near/far gap between members.

### Checked cheaply and not recommended
- Per-class output recalibration of the GRU: ≤ +0.0005 cross-fit (S10).
- Class-mix reweighting as a *selection* criterion: rankings move by ≤ 0.0006 (table 1.1). As a
  training weight it is already refuted (E-104).
- Per-sequence level shrinkage: the optimal between/within balance is about 1 (S8).
- Direct cross-sequence mixup of rows/targets: it breaks code-equality event features and the
  class-specific target lattice. MixStyle (M5) is the safe form.
- V-REx across chains (Krueger et al., arXiv:2003.00688 [abs]): equalizing per-sequence risk fights
  the pooled metric, where class difficulty differences are legitimate.

## 5. Selection-process recommendations

1. Add the M1 stress cells to every `decide.py compare`. For mask-related arms (weights, floors,
   blends, distillation) the primary cell is far-2/3 at λ .02/.05. For representation arms
   (M5/M6), also require a positive S9 residual.
2. Discount valid gains when forecasting LB: across 20 cached models, far = 0.72-0.79 × near. A
   near-valid gain of +0.001 is worth about +0.0008 on test-like sequences.
3. Log as ledger entries (for the lead): S2/S3 (mask is an absolute-activity rule; class 2 drives
   λ), S7 (LB-calibrated stress proxy), S9 (effective robustness), S14 (the ab25 cache is zeros).
4. Keep the adversarial model and its split frozen (hash the npz), like `splits-v1`. Refitting it
   on all train sequences is the one allowed change, and the §2 table must be re-checked after it.

## 6. Requests (bigger than my limits)

- **R1** (CPU, about 2 min): save OOF adversarial p(valid) per valid sequence; add `--stress` to
  `decide.py` (M1).
- **R2** (GPU, about 15 min each): M2(a) ε 0.1+aux, M2(c) randomized floor, M3 tilted distillation,
  M5 MixStyle, M6 (fix `predict_run.py`, re-score ab25, then variant S), M7, M8. Suggested order:
  M2a, M3, M5, M2c, M6, M7, M8.
- **R3** (CPU, about 10 min, ≤ 3 GB): the Gram cache for M4, K=6 members.
- **R4** (CPU scan): per-sequence stats for all 10.6k train sequences (`cache/seqstats_train.parquet`
  has 2,000), for M2(b) support scores and a stronger adversarial model for M1.
- **R5** (LB slot, human gate): eps_aux(train+valid) + 0.3·baseline, or the M3 student. The proxy
  predicts about 0.659 / 0.6586. A result ≤ 0.6569 would count against the stress proxy.

## Sources

Local PDFs (`~/Documents/Trading/`):
- Petrik, *Robust Reinforcement Learning* (tutorial slides), `RobustRL.pdf`: slides 3, 19, 54,
  64-66, 83-84.
- Charles & Darné, *Variance ratio tests of random walk: an overview*, J. Econ. Surveys 23(3)
  2009, `Heteroscedasticity.pdf`: §2.1, §2.3 eq. 10-13, §3.1 eq. 16-17, §4.1.
- Benidis, Feng & Palomar, *Optimization Methods for Financial Index Tracking*, FnT Optimization
  3(3) 2018, `Прочитать в 2025/Palomar_FinEng.pdf`: §3.3 eq. 3.21-3.25, §3.4 eq. 3.28, §3.5
  eq. 3.31-3.33, §5.4.
- Ayadi, Bouchard & Pascal, *Elliptical Wishart distribution: MLE from information geometry*,
  ICASSP 2023, `RobustEstimation.pdf`: §1-2, §4.
- Vertox, *Statistical Arbitrage in the U.S. Equities Market (But in Crypto, with code, Part 2)*,
  2023, `Naive/…Part 2.pdf`: p.1-10.
- Stat Arb, *Quant Roadmap 2024/2025 Ultimate Edition*, `RoadmapUltimateEdition.pdf`: ch. 13
  pp.34-35; ch. 19 pp.53-54.

Web (abstract-level unless noted):
- Sagawa et al., [arXiv:1911.08731](https://arxiv.org/abs/1911.08731) (group DRO, ICLR 2020).
- Zhou et al., [arXiv:2104.02008](https://arxiv.org/abs/2104.02008) (MixStyle, ICLR 2021).
- Tobin et al., [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) (domain randomization, IROS 2017).
- Wortsman et al., [arXiv:2109.01903](https://arxiv.org/abs/2109.01903) (WiSE-FT, CVPR 2022).
- Byrd & Lipton, [arXiv:1812.03372](https://arxiv.org/abs/1812.03372) (importance weighting in deep learning).
- Laroche et al., [arXiv:1712.06924](https://arxiv.org/abs/1712.06924) (SPIBB, ICML 2019).
- Miller et al., [arXiv:2107.04649](https://arxiv.org/abs/2107.04649) (accuracy on the line).
- Taori et al., [arXiv:2007.00644](https://arxiv.org/abs/2007.00644) (natural distribution shift / effective robustness).
- Krueger et al., [arXiv:2003.00688](https://arxiv.org/abs/2003.00688) (REx / V-REx).
- Yamada et al., [arXiv:1106.4729](https://arxiv.org/abs/1106.4729) (RuLSIF, Neural Computation 25(5) 2013).
- Zanini et al., [PubMed 28841546](https://pubmed.ncbi.nlm.nih.gov/28841546/) (Riemannian re-centering for BCI transfer, IEEE TBME 2018).
