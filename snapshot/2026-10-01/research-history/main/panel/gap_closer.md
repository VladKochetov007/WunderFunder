# Gap closer: what bought the last +0.3-0.5% elsewhere, mapped to our ledger (scout, 2026-09-27)

Question: which "free" variance or bias reductions can close the +0.0033 gap (LB 0.6618 vs 0.6651)
under 1 vCPU / 60 min, where a 3rd GRU member costs ~+12 min? Sources: past-competition write-ups
and papers only. Nothing about the current competition or its participants was read. The literature
is tier C evidence and each card's test is tier B. Offline gains transfer to the LB at 0.5-0.8x
(E-163, E-165). New numbers come from two read-only CPU checks (M1, M2). Their scripts are in the
session scratchpad (`pair_avg.py`, `holdout_mix.py`) and were run with `scripts/limited` at 6% CPU, 3G.

## 0. Verdict
- Top teams in market comps got their last and largest gains from **online learning on freshly
  revealed labels** (Jane Street 2024: +0.006-0.008 R²; Optiver 2023: retraining 5x). That lever does
  not exist here: there is no label feedback and no state across sequences, E-078/E-080 are closed, and
  its transferable part (training on the most test-like rows) already shipped as the valid FT.
- The rest of their last mile: **seed ensembles** (JS 2nd: 17 seeds; JS 6th: 6 models, +7% relative),
  **longer training** (JS 2nd: 5 to 9 epochs, +7% relative, better on all 10 time blocks), and
  **multi-head/multi-split averaging** (Wunder RNN 2025, 11th place: +0.003 R² each). All three are
  variance reductions, the same family as our EMA win.
- Brainteaser reading (M1): averaging two independent 2-epoch runs gains +0.0010-0.0012, so the
  infinite-seed limit is ~+0.002-0.0024. EMA-4 gained +0.0035 (E-164), more than any seed ensemble
  of the old recipe could. The binding variance is therefore mostly **label-noise memorization**, which
  limits epochs, and less **seed/SGD variance**. Levers that attack the first (cleaner targets, longer
  EMA runs) have more headroom than seed averaging. Seed averaging still has ~+0.001 per pair left.
- Every card below except C5 costs zero inference time. Honest total if C1-C3 all pass: about +0.0015-0.002
  on the LB, roughly half the gap. None of them reaches +0.003 alone.

## 1. New measurements (read-only, cached predictions)
| id | check | result | consequence |
|---|---|---|---|
| M1 | Average of two independent fold models, ab29_mask_f{k} (2 ep) + ab56_ema_f{k} (2 ep, EMA consistency), each scored on its own held-out chain fold | avg - mean(single): f0 +0.0010 / f1 +0.0010 (lam 0), +0.0011 / +0.0012 (lam 0.02); avg - best single +0.0001..+0.0005; residual corr 0.993-0.994 | Seed headroom ~+0.001 per pair, ~+0.002 at K = inf. It is smaller than the EMA-4 gain, so EMA did more than average out SGD noise |
| M2 | Holdout (368 seqs): EMA-4 final (ab60_..._lpft) vs 2-ep LP-FT final (ab51) and their average, alone and in the 0.65/0.35 baseline blend | ema4 0.6953, ab51 0.6926, avg 0.6948 (-0.0005); in the blend 0.6947 vs 0.6941 (avg) vs 0.6945 (0.75/0.25). Same order at lam 0.02; residual corr 0.995. s3ft+s4ft average +0.0007 over their mean | An extra GRU member must be an EMA-grade seed. Mixing in an older-recipe GRU loses |
| M3 | Per-epoch EMA eval (history.json) | f0 0.6641/0.6733/0.6757/0.6761; f1 0.6500/0.6592/0.6614/0.6623 | Not saturated at epoch 4 (+0.0004/+0.0009 in the last epoch) |

## 2. Claim-source pairs
| # | claim (source) | URL | maps to | status here |
|---|---|---|---|---|
| S1 | JS RTMDF, 6th private: 2 GRU archs x 3 seeds, LB 0.0105 -> 0.0112; aux responders +0.001; online update per date +0.008 CV | https://github.com/evgeniavolkova/kagglejanestreet (solution.md) | C5/C2 (seeds); E-107 (aux); E-078/E-080 (online) | seeds open; aux, online closed |
| S2 | JS RTMDF, 2nd (P. Yam, independent reproduction): 17 seeds, 9-target head, online lr 1e-4 +0.0064 R²; 5 -> 9 epochs +0.00136 R² on all 10 blocks | https://github.com/cweill/jane-street-causal-forecasting | C1 (epochs), C2/C5 | open |
| S3 | Optiver TATC, 1st: CatBoost/GRU/Transformer 0.5/0.3/0.2; retrain every 12 days (5x); zero-mean post-processing; "online learning and post-processing drove the score" | https://www.docswell.com/s/8980249862/K6YQ3E-2024-05-23-200638 ; https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution | family diversity (baseline member); E-160/E-109 (post-proc) | post-proc closed |
| S4 | Wunder RNN challenge 2025, 11th: 4 MLP heads averaged +0.003 R²; 3 models on different seeds and splits +0.003; LR schedule +0.005; deeper GRU, dropout, LSTM/TCN/attention and augmentation did not help | https://github.com/KlishevDA/WunderFund-RNN-Challenge-11th-Place-Solution | multi-head (table §5); failures match E-081/E-083/E-156 | multi-head low prior |
| S5 | Same challenge: a published 5-GRU seed/width ensemble, CV R² 0.389 single -> 0.3922, 1-core CPU 32 of 60 min | https://huggingface.co/msrishav/wunder-rnn-gru-ensemble | C5 | ~+1% relative from seeds |
| S6 | Same organizer re-scored finals "on a private fresh dataset" after the deadline | https://habr.com/ru/companies/wunderfund/articles/954662/ | audit H5; runtime margin for C5 | risk |
| S7 | G-Research Crypto: top 3 all LightGBM, feature engineering dominated | https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/ | E-088/E-089 (features flat) | not transferable |
| S8 | Numerai: models on other targets "can outperform or ensemble well with" the main-target model | https://docs.numer.ai/numerai-tournament/data | ML lead X2 (objective diversity) | needs a 3rd member |
| S9 | EMA of weights: better generalization, robust to noisy labels, less LR decay needed (Morales-Brotons 2024) | https://arxiv.org/abs/2411.18704 | E-154/E-164, C1 | supported |
| S10 | Weight averaging + LR annealing is best; averaging acts like a shorter decay (Ajroldi 2025) | https://arxiv.org/abs/2502.06761 | C1 schedule | - |
| S11 | EMA scaling rule: at batch kB use decay rho^k (Busbridge 2023) | https://arxiv.org/abs/2307.13813 | X5 (B128 x C500) + EMA | - |
| S12 | Distillation works when "patient and consistent": long schedules (Beyer 2022) | https://arxiv.org/abs/2106.05237 | C2; E-097/E-116 used 2 epochs | new evidence |
| S13 | Students often fail to match teachers (an optimization problem); for ensemble teachers, more fidelity gives better generalization (Stanton 2021) | https://arxiv.org/abs/2106.05945 | C2 fidelity readout | - |
| S14 | Under high label noise the optimal self-distillation teacher weight is > 1 (Das & Sanghavi 2023) | https://arxiv.org/abs/2301.13304 | C2; our distill used 0.5 | new evidence |
| S15 | The Bayes-distilled risk has lower variance than the empirical risk (Menon 2020) | https://arxiv.org/abs/2005.10419 | C2, C3 | - |
| S16 | Privileged intermediate time-series values give lower or equal risk at any sample size (a Rao-Blackwellized OLS in the linear-Gaussian case); nonlinear extension | https://arxiv.org/abs/2110.14993 ; https://arxiv.org/abs/2209.07067 | C3 = algebra card A (untested) | open |
| S17 | Codistillation: each copy matches the others' average prediction (Anil 2018); PAPA pulls a population toward its weight average, +0.8-1.9% vs independent models (2023); late-phase weights (2020) | https://arxiv.org/abs/1804.03235 ; https://arxiv.org/abs/2304.03094 ; https://arxiv.org/abs/2007.12927 | C4 | open |
| S18 | Nets become stable to SGD noise (linearly connected) after 3-20% of training (Frankle 2020) | https://arxiv.org/abs/1912.05671 | C4 branch point; E-156 (FT children too similar) | - |
| S19 | MIMO: several independent subnetworks in one forward pass (Havasi 2021) | https://arxiv.org/abs/2010.06610 | D3 (W_hh effective rank 16-32 of 128) | long shot |

## 3. Which variance each lever attacks, and runtime fit
| component | evidence it binds | levers | inference cost |
|---|---|---|---|
| label-noise memorization (limits epochs) | E-152/E-153, EMA-4 +0.0035 > seed limit (M1) | C1 longer EMA, C3 cleaner targets, C2 teacher targets | 0 |
| seed / SGD variance | M1 +0.001 per pair, residual corr 0.993 | C2 distil, C4 population, C5 member | 0 / 0 / +12 min |
| head / selection variance | E-148 (FT gain lives in the head) | bootstrapped multi-head (§5) | +1-3% |
| shift bias (new period) | S6, E-129 | baseline share, valid FT (shipped) | 0 |
TTA and "TBPTT-offset" ensembles each need one more full GRU pass per view, so they cost as much as a
member. Training already carries state from row 0 (`scripts/train_rnn.py:454`), so offsets would only
move the gradient-truncation points, which is free and worth about 0.

## 4. Hypothesis cards (ranked by E[ΔLB] per unit cost)

**C1. H-GC1 Longer EMA run with a stretched schedule (queued as tree F1; this sets its design).**
- Mechanism: EMA removes the memorization penalty, so extra epochs keep adding signal (M3 still
  rising). S2 found the same in a comparable streaming GRU setting (5 -> 9 epochs, every block improved).
- Change: 6 epochs, with OneCycle stretched over the whole run (not appended), EMA decay {0.999,
  0.9997}. 0.999 covers ~0.26 epoch at B32 x C2000 (~3.9k steps/epoch). Pre-register the final epoch.
  If X5 (B128) lands first, set decay to 0.999^4 ≈ 0.996 (S11).
- Test: folds 0/1, 4 runs. Falsifier: EMA-6 <= EMA-4 + 0.0003 at lam 0.02 on either fold.
- Expected: single +0.0004..+0.0010, LB +0.0003..+0.0007, P ≈ 0.5. Zero runtime.

**C2. H-GC2 Patient distillation of a K-seed EMA ensemble with teacher weight xi >= 1.**
- Mechanism: with a pure teacher target the student sees no label noise, so memorization no longer
  caps training length. It can train long (S12) toward the lower-variance ensemble mean (S15). At
  high noise the optimal xi is > 1 (S14).
- Not a re-run of E-097/E-102/E-116. Those used xi = 0.5, 2-epoch students trained partly on noisy
  labels, and a mixed-recipe teacher. M1 shows the seed headroom (+0.001 per pair) still exists.
- Step 1 (gate, also decides C5): 2 more EMA-4 seeds on fold 0 (seed 1 = ab56_ema_e4_f0). The
  held-out 3-seed ensemble minus the single must be >= +0.0008 at lam 0.02. If EMA already removed
  the seed variance, this kills both C2 and C5. The loader is host-bound and the GPU mostly idle
  (reflection 09-27), so train the K seeds as K modules on the same batches in one process for about
  one loader's wall time.
- Step 2: student target = xi*T + (1-xi)*y with T = teacher mean on the training rows, xi in
  {1.0, 1.3}. Keep the |y| p^2 weights: T depends on x only, so the weights do not move the minimizer.
  6 epochs with EMA; aux heads keep the true labels. Readout: fidelity corr(student, T) on held-out
  rows vs corr(single, T) (S13).
- Falsifier: student - single EMA-4 < +0.0004 at lam 0.02 on fold 0, or fidelity no better than one
  seed's.
- Expected: single +0.0005..+0.0010, LB +0.0003..+0.0007, P ≈ 0.35. Zero runtime.

**C3. H-GC3 Rao-Blackwellized (privileged-path) class-3 targets = algebra card A, upgraded.**
- Mechanism: condition y on the first k future book moves and integrate the rest with an x-free
  kernel. E[target|x] is unchanged and the noise variance falls (S16).
- New evidence for priority: M1 + E-164 show the binding variance is noise memorization. The ML
  lead's "variance not binding" argument against non-causal teachers predates E-164 and is
  contradicted by it. A causal teacher (C2) cannot remove label-noise realization; a privileged path
  can. Class 3 carries 46% of weight and has noise ratio ≈ 0.5 at k = 1 (card A): about a doubling of
  class-3 data, worth +0.002-0.003 per doubling for all data (E-152).
- Test: CPU gate first, with `scratchpad/alg/rb.py` (written, not run at scale) on ~200 class-3
  sequences; go if the noise ratio is <= 0.7 and the bias share is <= 0.05. Then one EMA-4 run per
  fold with class-3 (target, weight) swapped for (m2/m1, m1) on exact-law rows only.
- Falsifier: bias share > 0.1, or the arm is <= parent at lam 0.02 on either fold.
- Expected: single +0.0008..+0.0015, LB +0.0005..+0.0010, P ≈ 0.3. Zero runtime.
- Stacks with C1: cleaner targets make extra epochs worth more. Classes 0-2 need a learned
  look-ahead teacher (S16 nonlinear) and only if class 3 passes.

**C4. H-GC4 Population mean-teacher (codistillation / PAPA), shipped as one set of weights.**
- Mechanism: extend the winning mean-teacher term from "my own EMA" to "the average of K peers'
  EMAs" (S17), with a periodic PAPA pull toward the weight average.
- Branch: shared first ~0.5 epoch, i.e. after stability onset (S18) and before peers collapse into
  one solution, which is what sank the FT-children soup (E-156). The shared origin also shares the
  normalization buffers that broke E-119. Ship the weight average if the interpolation barrier ≈ 0,
  else one peer.
- Test: fold 0, K = 3, 4 epochs, same one-process trick as C2. Readouts: barrier; soup vs peer vs
  peer ensemble. Falsifier: soup <= best peer + 0.0003 at lam 0.02, or barrier > 0.002.
- Expected: single +0.0003..+0.0008, LB +0.0002..+0.0005, P ≈ 0.3. Zero runtime.

**C5. H-GC5 A 2nd EMA-grade seed as a 3rd package member, only after buying back runtime.**
- Evidence: S1/S2/S5 (seeds are standard last-mile); forensic C2 (+0.0005 inside the blend);
  E-146 (+0.0003 holdout, 3 members ≈ 50 min projected). M2: the extra member must be EMA-grade.
- Runtime: 36.5 + 12 ≈ 48.5 min against a 60-min cap, on a private set that may be a fresh period of
  unknown size (S6). Gate: bf16 weights (opus_arch D5: corr 1.00000; int8 moves WP up to 0.0006) plus
  fused/blocked GEMV (D8: 1.7-2.9x) bring the 3-member package to <= 40 min projected, and the C2
  step-1 ceiling is >= +0.0008.
- Falsifier: the 3-member blend is < 2-member + 0.0003 on the holdout.
- Expected: LB +0.0003..+0.0005, P ≈ 0.7. If C2 reaches its ceiling, C2 replaces C5 at zero cost.

Order: C3 CPU gate and C2 step 1 now (both cheap and decisive), C1 in parallel, then C2 step 2 on
the best C1 schedule (with C3 targets if they pass), C4 as the long shot, C5 only if C2 fails on
fidelity while the ceiling is real.

## 5. Considered and not ranked (mapped to the ledger)
| idea | top-team or literature support | our evidence | verdict |
|---|---|---|---|
| online learning / TTT on revealed labels | S1, S2, S3 (largest lever there) | E-078 (helps only at lag <= 50 rows), E-080 (rebuilt class-3 labels carry no new info); rules: no cross-sequence state | closed |
| metric post-processing (zero-sum etc.) | S3 | E-160, E-109, E-127, E-138 | closed |
| aux targets at other horizons | S1, S2, S8 | E-027, E-107 flat; aux-next already shipped | closed (SSL heads L11 remain) |
| multi-head output averaging | S4 (+0.003 R²) | linear heads average to one head under MSE; E-148 gain lives in the head | only bootstrapped MLP heads (per-head Poisson sequence weights); +1-3% runtime; low prior |
| MIMO subnetworks | S19 | spare capacity (D3 rank 16-32) is the precondition | long shot; +~15% MACs (input projection) |
| noise-aware (inverse-variance) row weights | heteroscedastic WLS | x-only weights keep the minimizer; E-104/E-138 mass/density weights flat; L8 untested | low prior |
| target smoothing over time | - | post-hoc smoothing refuted (timing M3, audit §5) | only as C3 |
| larger effective batch | S11 | X5 pending, E-123 chunk 500 weak | do with the EMA rule |
| family diversity member (gcorr, encoders) | S3, S8 | E-093/E-115, ab61 encoders at recipe level | ML lead X2/X8 |

## 6. For the controller's ledger (not written by me)
- E-166 (M1): average of two independent 2-ep fold models +0.0010-0.0012 over the mean single,
  residual corr 0.993 -> seed headroom ~+0.002 at K = inf, below the EMA-4 gain.
- E-167 (M2): on the holdout, averaging EMA-4 with a 2-ep LP-FT final hurts (-0.0005 alone, -0.0006
  in the blend). Extra members must be EMA-grade seeds.
- Hypotheses H-GC1..H-GC5 as above. C3 re-opens algebra card A with new priority.
