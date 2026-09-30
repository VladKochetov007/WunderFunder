# Specification critic, round 4: pre-registered rules for the next 8 hours (2026-09-27)

Scope: the 7 planned experiments and uploads. Read: contract, tree, reflection, panels, E-140..E-168, decision
scripts, trainer checkpoint/loss/label paths (no new-idea code, no compute). Units: pooled global WP on held-out
chain folds 0+1 (`fold_compare.py`), primary cell lam 0.02.

## 0. Facts that change the plan (fix before the first arm)
- F1 **Checkpoint = best epoch on the evaluation set** (`train_rnn.py:765-769`: held-out fold for fold runs, holdout
  for finals). EMA-4 curves were monotone (f0 .6641->.6761, f1 .6500->.6623, final holdout .6838->.6946), so E-164 is
  clean, but EMA-6, a 0.9997 teacher or a student may peak early. Save `last.pt` or assert argmax(history) == last.
- F2 **The EMA-4 fold refs have no cached predictions** (`cache/pred_ab56_ema_e4_f0_valid.npy`,
  `pred_ab58_ema_e4_f1_valid.npy`; the queued `pred` never logged). E-164's +0.0037/+0.0033 are `history.json` lam-0
  numbers without bootstrap, lam 0.02 or guards. Predict both and re-run `fold_compare` vs `ab46_ref` first.
- F3 **EMA-recipe seed noise was never measured.** `ab58_ema_e4_s4_f0` was queued (q18) but never ran. sigma_s =
  0.001 is a 2-epoch-recipe number (forensic C1, E-156).
- F4 **Every REC arm uses `--seed 3`**, which pairs nothing: any recipe change decorrelates the trajectory. Only
  same-trajectory readouts pair (epochs, eval-only EMA shadows, shared-parent FT; SE ~0.0001-0.0002).
- F5 **Fast path.** Fused AdamW is T1. Unless it is bitwise T0 with the ab56/ab58 path, re-run the seed-3 EMA-4 refs
  on it (the old-vs-new pair is a free seed-noise sample).
- F6 **Selection burn:** fold 0 selected E-152/153/154 and the E-168 screen, fold 1 was used once (E-164); folds
  2-3 never saw an EMA-recipe run and are the only clean confirmation folds.
- F7 **The trainer derives labels and weights from y.** `sequence_class(y)` (outcomes.py:31) infers the class from
  the quantum of y. `structure_labels`, `next_change_labels`, `shifted_targets`, the |y| weights and the loss
  normaliser (nn.py:110-113) also read y. E5 and E6 must pass new targets as extra columns and keep raw y elsewhere.

## 1. Evidence standard (applies to every experiment)
sigma_tot^2 = SE_boot^2 + sigma_s^2 (1/n_a + 1/n_r)/2, with n = seeds per fold, SE_boot = pooled 2-fold chain
bootstrap (~0.0006-0.0008), and sigma_s = 0.001 until Block 1 measures it.
**STACK** = posterior P(Delta>0) >= 0.8 under a sceptical N(0, 0.001^2) prior (zero-inference-cost levers only).
**CLAIM** = Delta >= 2 sigma_tot (any port, runtime cost, or ledger "supported"). Both need every fold > 0, lam 0 > 0
and stress lam 0.05 >= -1 SE.

| design (pooled folds 0+1) | sigma_tot | STACK | CLAIM | 80%-power MDE |
|---|---|---|---|---|
| 1 seed/fold vs 1-seed ref | .0010 | +.0012 | +.0020 | .0028 |
| 1 seed/fold vs 3-seed ref (after ladder) | .0009 | +.0010 | +.0018 | .0026 |
| 2 seeds/fold vs 3-seed ref | .0008 | +.0009 | +.0016 | .0023 |
| 2^3 factorial main effect (8 vs 8 runs) | .00055 | +.0006 | +.0011 | .0016 |
| same-trajectory readout (EMA shadow, epoch) | .0002 | +.0003 | +.0004 | .0006 |

All planned levers are predicted at +0.0003..+0.0015, so one-seed CLAIM power is 5-25%. Most verdicts will read
"unresolved at MDE X"; only Delta < -1 sigma_tot counts as refuted. Screening runs never count for the arm they
selected, and a best-of-~5 winner carried forward loses 1.0 SE from its forecast.

## 2. Per experiment: failure modes, design, pre-registered decision
**E1. EMA-6, OneCycle stretched over 6 epochs, decay 0.999 vs 0.9997.** Predicted Delta(EMA-6 - EMA-4) =
+0.0000..+0.0008 (median +0.0004); P(CLAIM) ~0.15.
- Failure modes: (a) Flat. M3's last-epoch gain (+0.0004/+0.0009) comes from the anneal to max_lr/2500, which every
  schedule has; both long-training recipes top out on fold 0 (E-153 wd 0.1: 0.6757; EMA-4: 0.6761).
  (b) The raw student memorises over 6 epochs at wd 0.01 and a 0.27-epoch EMA follows it (an E-153-style peak at
  epoch 4-5). (c) A 0.9997 *teacher* lags ~0.9 epoch (3,750 steps/epoch) and drags learning (E-154: -0.0011 at
  2 ep), so decay confounds teacher regularisation with the averaging horizon.
- Design: seed 3 per fold, teacher 0.999, plus an **eval-only 0.9997 EMA shadow**. The shadow has no feedback, so the
  trajectory stays T0 and decay is read at SE ~0.0002. Run a 0.9997-teacher arm only if the shadow wins by
  >= +0.0004 on both folds. Reference: EMA-4 seeds 3/4/5.
- Accept at STACK (>= +0.0010) only if the EMA curve drops <= 0.0003 over epochs 5-6 (else "unsaturated" is
  falsified). Kill at < +0.0005 or a negative fold: stay at EMA-4 (1.5x cheaper to train); 0.999 wins ties.

**E2. conv1d(k=16) encoder, fold 1 and EMA-4.** E-168: +0.0018 +- 0.0010 (bootstrap only, fold 0, 2 ep, best of 5-6
arms) is z ~1.3 with sigma_s; shrunk expectation +0.0005..+0.0008; residual corr 0.9956, so no diversity value.
- Failure modes: (a) regression to the mean; (b) EMA absorbs it, since both smooth high-frequency noise.
  (c) Train/stream mismatch: the causal conv needs the last 15 normalised rows carried across 2000-row training
  chunks, 4000-row eval chunks and the streaming engine; zero padding at chunk starts makes batch and stream
  disagree. (d) LP-FT's `--freeze inp,gru` leaves conv weights trainable unless their names match those prefixes.
- Design: skip the 2-epoch fold-1 run (wrong recipe). Run conv + EMA-4 on folds 0/1 and exclude the E-168 run.
  Pre-port gates: predictions identical (max abs < 1e-5) at eval chunk 2000/4000/20000 and in streaming; projected
  native runtime <= +2 min. Accept only at CLAIM (>= +0.0018); otherwise do not port.

**E3. Loss normalised by a constant instead of the per-chunk weight sum (card B).** Card: +0.0003..+0.0008.
- Failure modes: (a) `clip_grad_norm_(.., 1.0)` (train_rnn.py:720) scales high-W_b chunks back down (log the clipped
  fraction; above 30% the lever is partly a no-op). (b) Rescaling only the main term shifts main:aux:consistency,
  which keep their own sums (:697/:703/:714). (c) A *running* constant is non-stationary in epoch 0.
- Design, pre-registered: constant = mean W_b over the first 200 steps, then frozen; main loss only; EMA-4, folds 0/1.
  Accept at STACK **and** dense-tercile Delta (E-132 slicing) >= overall Delta; a gain without that concentration is
  noise. The effect is below every one-seed MDE, so test it only inside the factorial.

**E4. prop_v2 weights with EMA-4.** Prior E-155: +0.0010 +- 0.0010 at 2 epochs (f0 +.0015, f1 +.0005).
- Failure modes: (a) EMA absorbs part of it. (b) A sharper p changes mean p^2 and the ESS, which moves the effective
  floor share (eps 0.03); report sum w and (sum w)^2/sum w^2 against v1. (c) Hygiene: the fold-k prop model must be
  fitted without fold-k masks, and the final's `prop_v2_devb` without the holdout; verify both. (d) Mask
  specialisation (steep negative dDelta/dlambda), which the stress guard catches.
- Accept: precision-weighted with E-155 (sigma incl. seed ~0.0014), STACK = new pooled Delta >= +0.0008 with both
  folds > 0. Kill if the new Delta <= 0.

**E5. Class-3 label denoising (card A).** Predicted +0.0003..+0.0010 on EMA-4 (EMA already removes some noise).
- Failure modes: (a) Jensen bias: averaging future *mid moves* before applying the round/stop-loss law differs from
  averaging the *payoff*; use target m2/m1 with weight m1, built on clipped payoffs. (b) Kernel bias: an x-free
  kernel for moves k+1..4 shifts E[target|x] wherever x predicts later moves. (c) F7: a non-quantised target
  misclassifies class 3 and corrupts the aux labels. (d) The class-3 weight share changes (sum m1 != sum |y|),
  confounding the arm with class reweighting (flat/negative, E-104/E-127); renormalise per sequence.
- CPU gate: noise ratio <= 0.7, bias share <= 0.05 (>= 200 class-3 train sequences, out-of-sample predictions). Then
  EMA-4 on folds 0/1, held-out fold scored on raw `valid_targets.npz`. Accept at STACK with within-class-3 Delta > 0
  on both folds and classes 0-2 within +-1 SE; a gain only in classes 0-2 means reject.

**E6. Distil 2-3 EMA seeds, xi 1.0-1.3, longer training.**
- Failure modes: (a) In-sample teachers: predictions on their own training rows carry memorised noise, and xi >= 1
  imports it. Cheap check: the fold-0 EMA-4 model on fold 1 (in-sample) vs the fold-1 model on fold 1 (out-of-
  sample); a gap > 0.005 undermines xi >= 1. (b) Leak: fold-k students may only use fold-k teachers, never a final.
  (c) Schedule confound: the student trains 6 epochs, so its reference is **EMA-6 at the same seed and schedule**
  (E1), not EMA-4. (d) Gate: the 3-seed EMA-4 ensemble beats the mean single by >= +0.0008 pooled (M1: +0.0010
  at 2 ep; EMA may have eaten it), else kill E6 and the 3rd member. (e) xi = 1.0 first; 1.3 only after 1.0 passes.
- Accept: student - EMA-6 >= +0.0010 pooled, both folds > 0, >= 40% of the ensemble-minus-single gap captured, and
  held-out corr(student, T) > corr(single, T).

**E7. bf16 training.** No tensor gain over fp16 on sm_86, the scaler sync is already gone (fused AdamW), the
persistent small-h cuDNN path may not take bf16, and it has 3 fewer mantissa bits than fp16 (already off by 0.049
in E-157). Parity can only be T2 (a new draw). Rule: bench first; kill at < 1.25x with zero quality runs; else
non-inferiority on folds 0/1 (pooled Delta >= -0.0003, no fold < -0.0015, fp32-native corr >= 0.99997).

## 3. Schedule (serial, one job; EMA-4 ~14 min, EMA-6 ~21 min, including predict)
| block | runs | min | output |
|---|---|---|---|
| 0 | CPU + 2 predicts | 30 | F2 preds and lam-0.02 recheck of E-164; E6 memorisation gap; E5 CPU gate; F5 parity; E7 bench |
| 1 | 4 | 56 | EMA-4 seeds 4, 5 x folds 0, 1: sigma_s(EMA), 3-seed refs, E6 teachers and ensemble gate |
| 2 | 2 | 42 | E1 EMA-6 (teacher 0.999 + 0.9997 shadow) x folds 0, 1 |
| 3 | 14 (min 4) | 200 (56) | full 2^3 factorial {prop_v2, conv, card B} on EMA-4 x 2 folds, ladder = base cell (minimal: E4 + E2) |
| 4 | 2 | 28 | E5, if the CPU gate passes |
| 5 | 2 + teacher preds | 70 | E6 student xi 1.0, if the Block-1 gate passes |
| 6 | 2 (+ fold-2 prop model) | 35 | fresh fold 2: chosen stack vs EMA-4 base; pass at Delta >= 0 and >= 30% of summed fold effects |

Factorial, pre-registered: effect = mean(high) - mean(low) of per-run fold WP at lam 0.02, SE from the 7 df of fold
x effect interactions (extra base seeds only refine sigma). ~7.7 h preferred, ~5 h minimal; cut card B first, then conv.

## 4. When a candidate may be uploaded, and how to split the 5 daily slots
Eligibility (all required; the user uploads):
- U1: every new lever reached STACK (conv: CLAIM) plus its mechanism check; stacks of >= 2 levers passed Block 6.
- U2: pre-registered epoch count, final-epoch weights (F1), LP-FT unchanged with a freeze covering every non-head
  module, blend 0.65/0.35 never re-tuned on holdout (it favours the GRU, W6) or LB.
- U3: holdout pair against the LB package `final_ema4_lpft_base35` at the same blend: Delta(lam 0.02) >= +0.0012,
  Delta(lam 0) > 0, stress lam 0.05 >= -1 SE.
- U4: forecast = 0.65 x min(holdout blended Delta, 0.63 x pooled single-model fold Delta) >= +0.0008, where 0.63 is
  E-164's single-to-blend ratio. Log it in the ledger before upload with sigma = sqrt(SE_h^2 + 0.0007^2 + 0.0004^2).
- U5: streaming = batch (corr >= 0.99997, rows >= 99), prefix invariance, determinism, <= 20 MB, docker `--cpus 1`
  projected <= 42 min (private may be a fresh, larger set, S6).

Slots: **S1** the best eligible candidate. **S2** optional private hedge, a deterministic transform of the S1 GRU
(baseline 0.45, or WiSE 0.5 instead of LP-FT), only if within 0.0005 of S1 and >= 0 at stress 0.05; never pick
between S1/S2 on an LB gap < 0.0008. **S3-S4** only for later-eligible candidates (E6, E1 stacks), at most 2 per 4 h,
each after the previous readout updated the transfer factor (0.65; refit when |z| > 1). **S5** held until 22:00 for a
failed/timed-out re-upload. An unused slot costs nothing; a forecast below +0.0008 only measures noise.

After each readout: z = (LB - forecast)/sigma. New incumbent only at posterior P(Delta>0) >= 0.9 (precision-weighted
offline + LB); two consecutive |z| > 2 stop uploads for a proxy re-audit. Private picks: best posterior plus one
hedge, never the raw LB maximum (best of 5 at SE 0.0004 is inflated ~0.0005).

## 5. What would change these rules
sigma_s(EMA-4) <= 0.0004 in Block 1: shrink Section 1 thresholds ~25%. E6 memorisation gap ~0: allow xi = 1.3 at
once. Block 6 < 0 for a stack that passed folds 0/1: confirm on folds 2/3 from then on. LB beats forecast by z > 2 on
a same-parent pair: raise the transfer factor, never pick levers from it.
