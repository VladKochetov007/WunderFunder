# Research plan, status and directions (night of 2026-10-09)

Root (Claude Code) note. Lane status at about 18:45 UTC, results so far, research directions with decision rules, and risks.
Companions: [optimizers and schedules](OPTIMIZERS_SCHEDULES_STABILITY_R1.md),
[interpretability and blend theory](MECHANISTIC_INTERPRETABILITY_AND_BLEND_THEORY_R1.md),
[telemetry design](TRAINING_TELEMETRY_AND_CALLBACKS_R1.md).

## 1. Lane status

| Lane | State | Notes |
|---|---|---|
| Vast (1 receipt-owned 2x RTX 4090, $0.88/h, about $4.5 spent) | 4 jobs running at 93-95% GPU, 15 queued after the new wave | 14 jobs finished, 0 failed. Tick check every 5 min pulls results; spend cap 12 USD only holds queued jobs (never kills running ones); no hard stop |
| Local GPU (one worker, 5 GB VRAM) | `h230-f1-s4-mexact-fit-r2` running, 17 items queued | admission barrier was closed 17:13-18:23 UTC after a supervisor RAM kill (my concurrent analyses), reopened with a written review |
| Kaggle | GPU idle: campaign GPU quota exhausted (57,341 s used of the 56,780 s campaign cap; platform 94,389 of 108,000 s) | capacity V5R2 kernel ended ERROR (both arms hit the 15,500 s worker timeout in epoch 3 of 4; epoch 0-2 checkpoints collected and queued for local scoring); native-TPU v4 kernel also ended ERROR (cause not yet diagnosed); rolling small-screen sampler paused by STOP until 03:05 Kyiv |
| Watchdog | `wnn-watchdog.timer` every 3 min, alert log, desktop notify, session monitor | first pass found the two failures above |

## 2. Results so far (fold1 unless stated, wp = `global_wp_clip2_v1`)

Scratch teachers (Muon, one-cycle, GRU256 unless named; seed noise about 0.001; scratch baseline 0.6662 from seeds 0.66557 and 0.66675):

| Model | wp alone | best blend gain over the s4 student (0.66913) |
|---|---|---|
| gated | 0.66867 | +0.0019 |
| GRU384 fixed_mix | 0.66817 | +0.0033 |
| GRU384 (seeds 4 / 3 / 5) | 0.66802 / 0.66724 / 0.66670 | +0.0032 / +0.0029 / +0.0023 |
| GRU256 seed 5 / seed 4 / 3-layer / trapezoid | 0.66778 / 0.66675 / 0.66774 / 0.66744 | +0.0010 to +0.0015 |
| GRU320 / **GRU448** / GRU512 | 0.66664 / 0.66565 / 0.65800 | +0.0017 / **+0.0034** / +0.0019 |
| wmse_gcorr / twoclock / LSTM256 / MLP encoder / D2Z / fixed_mix256 | 0.66625 / 0.66583 / 0.66581 / 0.66528 / 0.66504 / 0.66490 | +0.0003 to +0.0009 |
| dual / AdamW (lr 1e-3) | 0.66275 / 0.66231 | +0.0001 / -0.0005 |

Findings, with evidence level:
- Every scratch recipe variant is within about 0.002 of the baseline or worse; AdamW is clearly worse (-0.004, LR untuned). [measured]
- The blend gain of a wide scratch teacher over the best student replicates across seeds on fold1 (GRU384 +0.0023..+0.0033, GRU448 +0.0034) and is larger than for width-256 seeds (+0.001). [measured, one fold]
- It did not transfer to fold0 with frozen weights (-0.0020 at weight 0.4) because the fold0 student leads the scratch teacher by 0.0122 (fold1: 0.0019). The pair-blend closed form predicts exactly this. [measured]
- Class 3 sequences carry 50% of the metric weight at wp 0.44 (others 0.74-0.81); class-gated blending and monotone recalibration gave nothing. [measured]

## 3. Research directions

| # | Direction | Why | Queued now | Decision rule |
|---|---|---|---|---|
| A | Diverse, near-peer blend partners | gain needs corr < r_j/r_i with a small quality gap | fold1 widths 320/448/384 seeds, gated384, gated plus trapezoid, class-3 specialist | keep a partner if its predicted two-model gain (closed form) is above +0.001 on at least two folds |
| B | Four-fold evidence | one fold cannot validate a blend rule; also needed for nested-CV stacking | scratch GRU384 and GRU256 on folds 2 and 3, fold0 GRU256 and 6-epoch GRU384 | width effect counts only if positive on folds 1, 2, 3 and not negative on 0 |
| C | Close the quality gap of wide partners | fold0 scratch lags the distilled lineage by 0.012 | 6-epoch fold0 GRU384 (queued); next: Vast-hosted distillation pipeline (see section 4) | a partner with gap under 0.004 to the student on fold0 |
| D | Training dynamics and schedules | budget, LR, optimizer, cyclic snapshots | telemetry waves (tr_base, wsd, ep6, lr, swa, b16), SGDR 2/4/8 with snapshot exports, optimizer grid (AdamW 3 LRs, Muon scope) | see OPTIMIZERS_SCHEDULES_STABILITY_R1 section 7 |
| E | Weight space | same-parent students may average into one model | `soup_f1_s3_a` scoring queued locally | soup beats best member by more than noise, or matches the prediction-space blend |
| F | Exact teammate-structure replication | the single strong teacher at 0.30 and seed-specific parents | local `mexact` s3 done, s4 r2 running, exports queued | compare with H230 s3/s4/pair (0.66882, 0.66913, 0.66946) |
| G | Interpretability and complementarity | which architectures encode what; CKA vs blend gain | to build: hidden-state extractor, gate timescales, probes | CKA/probe distance predicts pair gain better than prediction correlation |
| H | Capacity (Kaggle) | GRU512/768 | partial epoch 0-2 checkpoints being scored | GRU512 already lost 0.0082 at 4 epochs on fold1; only revive if epoch-2 fold0 results are strong |

## 4. Next builds (in order)

1. **Vast-hosted student pipeline** (direction C). Steps on the host: export train-split predictions of selected scratch teachers with `predict_run.py`; build a heterogeneous distillation cache with `build_distill_cache.py` (copy that script and `src/wnn/distillation.py` into the bundle); fit a warm-started student with the H230 recipe on the same box. This tests whether wider and more diverse teachers improve the *student*, which is the original teacher-quality question, without moving the large caches over the slow laptop uplink.
2. **Nested-CV stacking tool** once folds 2 and 3 have teachers: fit blend weights on three folds, score on the fourth, report against the best single and against the closed-form rule.
3. **Hidden-state probes** (gate timescales, linear probes for market class and time-since-event, CKA) on GRU256, GRU384, GRU448, gated, LSTM.
4. **Diagnose the TPU v4 error** from collected outputs, then decide whether TPU work is still worth the quota.

## 5. Risks and guards

- Aggregate local RAM: the supervisor kills the whole fit at 18 GB. Root analyses run only with explicit `scripts/limited` limits; the grading service is capped at 3.5 GB.
- Kaggle GPU quota is exhausted for the campaign; the rolling sampler resumes only after the quota guard allows it. CPU-only Kaggle kernels remain available for cache building.
- Vast: one instance, receipt-owned; the watchdog reports a dead instance, ssh loss, an idle GPU with a queue, a dead refill daemon, stalled or failed jobs.
- Selection on exposed folds: every blend weight chosen on a fold is optimistic; report transfer to a different fold before believing a gain.
