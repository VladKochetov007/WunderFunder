# Game changers: what moved the LB, and new ideas of the winning types (2026-09-27)

Strategist report (conduct-creative-research, tier B). Inputs: reflection/findings/plan 09-26, tree, panel/*, E-060..E-169.
New CPU probes today (read-only, `scripts/limited` 6% CPU, 1 thread, 4G), on cached OOS predictions, each model scored
on its own held-out chain fold: session scratchpad `gc/m1_blend.py`, `m1_pooled.py`, `m3_ftgain.py`, `m4_maskinfo.py`,
`m5_swap.py`. Unit: pooled global WP. λ = weight on unscored required rows. The primary cell is λ 0.02 (critic_round4).

## Part 1. Taxonomy of the LB gains, checked against the ledger

| type | LB steps | offline evidence | mechanism | why it bound here | transfer |
|---|---|---|---|---|---|
| **A** exact metric/scorer alignment | 0.5935→0.6503 (+0.0568, baseline x0.25); wMSE inside 0.6548 | E-061 +0.0557 (global); E-071 wMSE vs CE +0.0162, Huber +0.0095, corr-mix +0.0035; E-072, E-160 | remove a deterministic mismatch between what we output or optimize and what the scorer computes (clip ±2, \|clip y\| weights, global pooling) | 39% of baseline outputs lay beyond the clip; METRIC.md described a per-sequence metric that the LB does not use (E-161) | ~1:1 (no estimation) |
| **B** train where the scorer looks | 0.6569→0.6586 (true masks, part of +0.0017); 0.6586→0.6600 (+0.0014, valid FT); 0.6601 (LP-FT +0.0001) | p² weights +0.034 per-seq (E-038/039) and +0.053 global (E-069); true masks +0.0006..10 (E-098); FT +0.0016±6 (E-124); prop_v2 +0.0010±10 (E-155) | spend capacity on the causal activity-selected ~9% scored rows. Exact masks exist only on valid, hence the FT | at equal propensity, scored rows are +0.15..0.22 WP more predictable (E-162) | ~1:1 (FT λ0 +0.0011 → LB +0.0014) |
| **C** variance reduction (label noise, little data) | 2 ep + train+valid inside 0.6548; floor + aux-next inside 0.6586; 0.6601→0.6618 (+0.0017, EMA-4) | 2 ep +0.0038 (E-083); floor +0.0020, aux-next +0.0014, tv-OOF +0.0014 (E-097); EMA-4 +0.0037/+0.0033 (E-154/164); WiSE halves FT variance (E-147) | lower the variance of the fitted function: averaging (EMA, blend), early stopping, cleaner auxiliary labels, more rows | ~30 label episodes per class-3 sequence (phys F3); +0.0015..34 per data doubling (E-152); memorization caps epochs (E-153); EMA beat the infinite-seed limit (E-166) | 0.5-0.8x |
| **D** independent error source | 0.6548→0.6569 (+0.0021, blend with baseline) | E-084 +0.0017±5; stress +0.0018 (E-111/113); gain in all classes (E-126), larger on train-unlike/calm (E-129/130) | average with a model whose errors are not ours (residual corr 0.956 vs ≥0.99 among our variants) | our variants are near-copies (E-111/115/120/121) | ~1x at λ .02 (P3) |

Corrections to the proposed taxonomy:
- **B is the second-largest win, and the LB history hides it.** p² propensity weights (+0.05 offline) sit inside every GRU we
  submitted. Without them no GRU would have beaten the scaled baseline (ab18 0.6488 already had them).
- **D is not "a different objective".** The baseline's objective and view (a raw-only plain-MSE GRU) add only +0.0002 (E-125), and the raw view explains
  less than half of the diversity (E-120). The source is unidentified. **D is shrinking:** at EMA-4 the OOS-optimal baseline share is 0.20-0.25
  (O1), and a second EMA-grade GRU beats the baseline as the second member (O5). C and D substitute for each other: much of the blend's value was
  averaging away our GRU's variance.
- The floor, pow2 and aux-next are C inside B/A. Sharper weights help until the ESS collapses (pow3 −0.002, E-048; ε 0.10 flat,
  E-102). aux-next is a lower-noise label: the side of the next change is the winner on 99% of rows (E-162).
- Meta-type with no LB delta that protected every later step: evaluator repair (chain folds E-095/096, seed-aware rule).

Failure families (the contrast):
| family | ledger | result | why it failed here |
|---|---|---|---|
| features / information | E-088/089/092/093/096/112/149, E-038/118 | ≤0; GBM gain was a leak | the GRU already tracks every main effect (E-106; symbolic overlap 0.85-0.92, E-100/101). Far class-3 rows are a coin flip for every signal (phys F4). Noise binds, not information |
| capacity / architecture | E-081/083/123/168, E-028 | flat, LSTM −0.0023 | data-limited: extra capacity memorizes race noise (E-153: 12 ep wd .01 peaks at ep 5, then −0.027) |
| post-hoc calibration | E-109, E-160 (15 maps ≤ +0.0007), E-161/162 smoothing, hazard rescale | ≤ +0.0007, class 3 < 0 | wMSE already outputs the Q-conditional mean (slopes .95-1.0, martingale increments +0.008). A fitted map only adds noise, and the class-3 map drifts across chains |
| per-segment weights | E-104, E-127, E-138, E-102, E-107 | −0.0004..−0.0027 | the metric is one pooled Q-MSE. Segment weights only move capacity, and segment-fitted parameters rescale segments, which pooled Pearson charges for |
| online recalibration | E-078, E-080 | −0.015..−0.025 | no test labels; lagged rebuilt labels add noise to an output that is already E[y\|F_t] |
| averaging without a noise regime | E-052 SWA, E-116 distill, E-119/156 soups, E-154 EMA at 2 ep | flat or − | variance reduction pays only where memorization binds (≥4 epochs). A same-time teacher adds no information beyond the student's row |

Rule that separates the columns: a win either removes a deterministic mismatch with the scorer (A, B) or lowers estimation
variance (C, D). A loss either adds information the GRU already has, or fits new parameters to noise at the segment or output level.

## Part 2. Measured today (offline, OOS; details in the cards)
| id | probe | result |
|---|---|---|
| O1 | EMA-4 blend geometry, folds 0+1 (ab56/ab58 + baseline x0.25) | optimum GRU share 0.79-0.86. vs w .65 at λ .02: w .75 **+0.00042±0.00020** (P>0 .99), w .80 +0.0005±0.0003; EMA-4 LP-FT f0 +0.00034. w .55 (critic's "baseline 0.45" hedge) **−0.0008±0.0002**. Single GRU ≈ the .65 blend (+0.0001±0.0006). Per-target weights disagree across folds (t0 .98/.79, t1 .74/.80): closed |
| O2 | per-target metric Jacobian c_k ∝ ρ_k/(Var_Q(p_k)·W_k) | c0/c1 = 1.01-1.08. The loss's equal target weights are within 8% of metric-optimal: **closed** |
| O3 | FT gain by OOS prop_v2 band (hybrid attribution, λ0) | 2-ep LP-FT (+0.0013±0.0008): **~90% of the gain comes from p ∈ [0.02, 0.35)** (30% of scored weight). The p ≥ 0.5 band (56% of weight) gives −0.0001±0.0005. EMA-4 LP-FT f0 (net +0.0001): **classes 0-2 +0.0012, class 3 −0.0012**. At λ .02 the p < 0.01 band loses −0.0002 |
| O4 | does the mask depend on model-visible structure beyond prop_v2? (HGB, chain halves) | AUC 0.932/0.931 → **0.937/0.936** with the OOS EMA-4 outputs; top quintile 0.776/0.777 → **0.794/0.793** (f0/f1), every class. \|D\| alone carries it (f0 +0.0043 / top +0.0146); \|ΔD\| adds 0; the organizer baseline's \|D_B\| (not trained with our weights) adds +0.0011 / +0.0043 |
| O5 | 2nd member: EMA-grade GRU vs baseline (fold 0: ab56_ema_e4_f0 + ab62_p2_f0, a decorrelated run, not a pure seed) | at λ .02: pair average 0.65658 vs the incumbent structure (.65 blend) 0.65526 (**+0.0013**) and vs the best baseline blend (w .80) **+0.0006±0.0007**; adding the baseline to the pair: −0.0001. Pair average − mean single = +0.0009 |

## Part 3. New ideas, ranked by expected LB gain per GPU-minute
Thresholds follow critic_round4 §1. Shared-parent LP-FT with 2 seeds per fold: σ_tot ≈ 0.0008, STACK +0.0009. EMA-4 arms with 1 seed per fold:
STACK +0.0012. Deterministic offline transforms: SE_boot only, STACK at Δ ≥ +0.0003 with P>0 ≥ 0.8. Every lever also needs each fold > 0,
λ0 > 0 and stress λ .05 ≥ −1 SE. Upload only if the U4 forecast is ≥ +0.0008. None of these duplicates an ideas_round4 card or the running wave (links noted).

| # | idea | type | GPU-min | P(pass) | LB if pass | E/GPU-min |
|---|---|---|---|---|---|---|
| 1 | blend weight .75 for EMA-grade GRUs (O1) | A/D | 0 | 0.8 | +0.0003 | ∞ |
| 2 | member swap: 2nd EMA-4 seed replaces the baseline (O5) | C vs D | 0 to measure, ~20 to adopt | 0.5 | +0.0005..10 | 2e-5 |
| 3 | blend-through LP-FT, zero code | D×B | 10 | 0.25 | +0.0002..6 | 1e-5 |
| 4 | closed-form last-layer lab | C×A instrument | 6 + 2 h code | 0.3 | +0.0001..4 | 1.5e-5 (+ makes 3/5/6 free) |
| 5 | class-3-anchored LP-FT (O3) | B×C | 20 | 0.35 | +0.0003..8 | 1e-5 |
| 6 | λ-matched FT floor, zero code | B×A | 10 | 0.3 | 0..+0.0003 | 6e-6 |
| 7 | prop_v3: weights sharpened by teacher \|D\| (O4), gated | B | 28 | 0.15 | 0..+0.0004 | 2e-6 |
| 8 | metric-weighted mean-teacher consistency | C×A | 28 | 0.2 | 0..+0.0004 | 2e-6 |
| 9-11 | O4 mask probe, O3 FT attribution, O2 target Jacobian | B, B, A | 0 (done) | - | decision value | - |

**1. Re-derive the blend weight for each GRU generation (A/D; 0 GPU).** Mechanism: each C-win shrinks our GRU's variance, so the
baseline's averaging value falls and the optimal GRU share rises (0.65 was fitted on 2-epoch models, E-131). Change: pre-register
w = 0.75 in the next EMA-grade package; never use the "baseline 0.45" S2 hedge (−0.0008, P>0 0.00). Rule: argmax at λ .02, pooled over
OOS folds 0/1, rounded to .05 and shrunk halfway to the previous w. Read-out met (+0.00042±0.0002). No LB slot on its own: the forecast
(+0.0003) is below the noise bar.

**2. Member swap (C displaces D; 0 GPU to measure).** Mechanism: EMA-4 kept the seed-pair headroom (+0.0009 over the mean single) but
cut the baseline's value to ≤ +0.0005. Our GRU is also cheaper at inference than the baseline member (~31-42 vs ~52 µs/row, E-085/146), so
runtime does not rise. Test: when the wave's `ab62_base_s4_f{0,1}` land, run `m5_swap.py` with ab62_base_s3/s4 on both folds. Pre-reg:
pooled Δ[avg(s3,s4) − (s3 + base, w .80)] ≥ +0.0006, both folds > 0. Upload if Δ vs the incumbent structure (w .65) is ≥ +0.0012. Adopt with one
final EMA-4 seed-4 + LP-FT run, and keep the baseline package as the private hedge (the baseline's gain grows with train-unlikeness, E-129).

**3. Blend-through LP-FT, "complement head" (D×B; zero code).** Mechanism: re-fit the head so that the *blend* minimizes wMSE on true-mask
valid rows, where the baseline is OOS. The identity a·out + (1−a)B − y = a·(out − [y − (1−a)B]/a) means the existing distill path
already computes this target: LP-FT recipe + `--distill baseline*0.25 --distill-alpha -0.333` (a = .75). Test: parents ab56/ab58, folds 0/1,
2 seeds, vs ab63_ctrl. Pre-reg: blended at a vs ctrl blended at its best w ∈ {.65,.75,.80}, STACK +0.0009. The single-model Δ is expected
< 0. Moot if #2 removes the baseline.

**4. Closed-form last-layer lab (C×A instrument).** The FT gain lives in the head (E-148), and FT-child seed noise (0.0008, E-156) swamps
its levers. One GPU forward per model dumps per-sequence Gram sums of φ = GELU(head.0([o,z])) (129-d) under the weights |y_k|·s, |y_k|·(1−s) and
p², plus cross-moments with y, the baseline and the parent. Exact WLS and ridge last layers and their held-out WP then become CPU solves:
ridge-to-parent path (a continuous WiSE), #3, #5, #6, per-class last layers (A3), chain bagging. No seed noise, so STACK is about +0.0006.
Validity gate: the lab reproduces ab63_ctrl_f0h within 0.0003. Lever gate: a CV ridge head beats the SGD LP head by ≥ +0.0004 on folds 0/1.

**5. Class-3-anchored LP-FT (B×C; ~15 lines).** Mechanism (O3): the FT adds information where the soft weights miss the scorer's
selection (classes 0-2, moderate p). Where p² already matches the scorer (class 3: prop AUC 0.94, 56% of weight at p ≥ .5) it adds only
variance. Change: `--anchor κ --anchor-classes 3`, with a frozen copy of the --init model and the term κ·Σ main_valid·(out − parent)² on class-3 rows; κ ∈ {1, 4}.
Test: folds 0/1 × 2 seeds vs ab63_ctrl, read with `m3_ftgain.py`. Pre-reg: STACK +0.0009, class-3 Δ ≥ 0, classes 0-2 Δ ≥ −1 SE. The class split
was selected on folds 0/1, so confirm on fold 2 before stacking. A3 (ideas_round4) is the +1-head Rust variant.

**6. λ-matched FT floor (B×A; zero code).** Mechanism: `--mask-weight` gives unscored rows weight 0, while the LB behaves like λ ≈ .02
(skeptic); O3 shows the FT loses on p < .01 rows at λ .02. Flags: replace `--mask-weight` with `--propensity prop_v1 --prop-power 2
--prop-eps 0.03 --valid-true-mask 0.02` in both LP-FT stages. Pre-reg: λ .02 STACK +0.0009 with λ0 ≥ −1 SE; report dΔ/dλ. Run as a 2×2 with #5.

**7. prop_v3 (B; ~30 lines; gated).** Mechanism (O4): the scorer oversamples states that a label-trained model finds directional, and prop_v2
misses this (top-quintile AUC +0.015..0.020). Change: multiply the weights by T(p_v2 bin × |D_EMA| bin) = calibrated E[s|p,|D|]²/p², fitted on OOS valid
folds and applied online with the EMA teacher's |D|. The factor is x-measurable, so the minimizer is unchanged; renormalize per batch. Gate: run only if the wave's prop_v2
EMA-4 arm is > 0 pooled (fold-0 early read −0.0002 at λ0). Pre-reg: STACK +0.0012, with the class 0-1 gain ≥ the class-3 gain.

**8. Metric-weighted consistency (C×A; 2 lines).** Mechanism: `cw = main_valid` carries no |y|, so the per-row pull toward the teacher is
κ/(|y|+κ). It is strongest on |y| ≈ 0 rows that the metric ignores (class-3 touch rows, 36% of t0) and weakest on the heavy rows where memorization acts.
Change `--cons-weight metric`: cw = |clip y|·main_valid per target, a uniform pull κ/(1+κ). Test: EMA-4 f0/f1 vs ab62_base_s3. STACK
+0.0012 and the epoch curve still rising at epoch 4. It stacks with TD (B1b).

**9-11. Done.** O4 feeds #7, O3 feeds #5/#6 and explains why LP-FT fades at EMA-4 (net +0.0001 on f0), and O2 closes per-target loss weights.

Order: #1 now (next package). #2 as soon as the s4 fold predictions exist. #3 + #6 + #5 as one LP-FT batch (~40 GPU-min, 2 seeds), or
free on CPU once #4 exists. #7/#8 only after the wave's TD/prop_v2 reads. Ledger rows for the controller: E-170 O1+O2, E-171 O3,
E-172 O4, E-173 O5 (fold 0 only; replicate on fold 1 before any claim).
