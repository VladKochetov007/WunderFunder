# Reflection after 7 leaderboard submissions (2026-09-26 15:40)

Metric: global pooled clipped |y|-weighted Pearson (mean of t0/t1). Leaderboard dense: rank 31 at
0.6600; top 0.6651; 70 teams inside 0.6573-0.6651. Scorer: 1 vCPU, 60 min; our 2-member native
package runs 36.3 min on the scorer (so ~23 min headroom; a third GRU member of the same size costs
~+6-8 min).

## Leaderboard history and attribution

| submission | LB | step | lever | where the idea came from |
|---|---|---|---|---|
| organizer baseline | 0.5935 | | | |
| baseline x0.25 | 0.6503 | +0.0568 | output scale vs clip at +-2 | own metric anatomy (E-060/E-061) |
| ab18 GRU wMSE | 0.6488 | (-0.0015) | single GRU, train only | local RNN experiments |
| final_tv_e2 | 0.6548 | +0.0045 vs base | 2 epochs, train+valid, |y|-weighted MSE | local RNN experiments (E-071, E-083, E-085) |
| blend_base_final_tv | 0.6569 | +0.0021 | 0.5 GRU + 0.5 baseline x0.25 | standard ensembling; baseline is the only diverse member |
| final_s4_base35 | 0.6586 | +0.0017 | propensity^2 weights with floor 0.03 (E-097), next-target-change aux head (E-097; haiku H2-4 "time-to-next-move"), true valid masks as weights (E-098; Opus scout M1), blend 0.65 (E-129/E-131) | mixed: own + haiku + Opus scouts |
| final_s4ft_base35 | 0.6600 | +0.0014 | 1-epoch fine-tune on valid dev rows with true masks, lr 2e-4 (E-124/E-141) | own follow-up of Opus M1 |

Totals over the organizer baseline (+0.0665): metric understanding 85%, recipe/data 7%, blending 3%,
mask-robust training 5%. Haiku random-literature rounds (4+4+1 files, ~100 ideas): one idea reached the
submission (aux head, +0.0014 on valid); every feature idea was refuted (E-088, E-089, E-093, E-112).
Opus scouts: true masks (+0.0006-0.0010), stress-panel proxy (selection), chain leak discovery
(validation correctness), shift diagnosis. Web/arXiv reading: validation methodology (chain-grouped
folds, bootstrap by chain) - no score directly, prevented false positives (old dev halves leaked).
There was no formal bandit policy: arms were run as sequential A/B with a decision rule (E-084/E-097).

## Proxy calibration (predicted vs LB)
| pair | proxy | LB |
|---|---|---|
| base - ab18 | stress +0.0016 | +0.0015 |
| final_tv - base | stress +0.0035 | +0.0045 |
| blend - final_tv | stress +0.0039 | +0.0021 |
| s4_base35 - blend | analog stress +0.0025 (+true masks) | +0.0017 |
| s4ft - s4 | folds ~+0.0005, leaky holdout +0.0009 | +0.0014 |
Sign right 5/5; magnitude error ~+-0.001-0.002. Differences < 0.002 are not resolvable offline.

## What we know (short)
- Target law: t0 = F - ask, t1 = bid - F, common future price F; exact for class 3 (large tick, 46% of
  weight, GRU 0.43-0.48 within class); approximate for classes 0-2.
- Train and valid are different instruments/regimes (adversarial AUC 0.99 within class); sequences are
  20k-row chunks of 3,288 continuous streams.
- Saturation: width/depth/LSTM/epochs>2/114 heuristics/GBM/online recalibration/seed blends all flat.
  Our GRU variants have residual corr 0.99+; the organizer baseline 0.956.
- Densely scored sequences carry 54% of metric weight and every model is ~0.14 worse there (E-132..E-139).
- Baseline blend gain grows with train-unlikeness and in calm sequences (E-129, E-130).
- NEW (E-144): fine-tuning on valid-like data helped the test ~3x more than valid folds predicted.

## Open questions for the panel
1. Is the test closer to valid than to train? What follows (valid oversampling, domain weighting,
   longer fine-tune, fine-tune on all valid incl. holdout, importance weights for train sequences)?
2. Where can +0.005 come from: scaling (model/data/compute), better learning (schedules, EMA/SWA,
   longer sequences), augmentation (bid/ask mirror symmetry with t0<->t1 swap, time warping, crop),
   filtering (drop train sequences unlike valid), a genuinely different model family, or the dense-
   scored regime?
3. How should the research process change (idea sources, evidence standard, proxy, LB use)?

## Update 2026-09-27 02:40 (LB 0.6618, rank 18; top 0.6651)
Trajectory: 0.6503 (scale) -> 0.6548 (recipe) -> 0.6569 (blend) -> 0.6586/0.6600 (mask levers, FT)
-> 0.6601 (LP-FT) -> 0.6618 (EMA mean-teacher, 4 epochs). The last two same-parent or recipe steps
transferred at 0.5-0.8x of offline estimates. Remaining gap +0.0033 = about one EMA-sized lever.
Brainteaser reading: we sit in a label-noise-dominated, data-limited regime; every recent win is a
variance reduction (true masks, fine-tune on scored rows, EMA). The panel found no metric free lunch.
Candidates in that family: sharper selection weights (prop_v2), loss normalization matching the
pooled metric, class-3 label denoising, distilled ensembles, and a genuinely diverse member.
Throughput: training runs at ~15 seq/s with the GPU mostly idle (DutyCycle sleeps + host-side
work + unpinned 287 MB batch copies); a verified speedup multiplies hypotheses per hour.
