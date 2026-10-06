
## Assistant (Build · Muse Spark 1.3 Free · 28.1s)

_Thinking:_



# Second look: where the campaign stands, what was mis-read, and what to do next

## 1. Current state in one paragraph

Public best is **0.6660** — complete combo seed 4 (`c20261004_combo_s4_i16.zip`, 37m07s/626MB, `2e5j6bnt`). Seed-3 complete is 0.6655, seed-3 head-only 0.6654, seed-3 parent-only 0.6641, old D256 0.6641. Decomposition (`PUBLIC_RESULTS_REFLECTION_20261004_LATE.md`):

```
parent_s3 0.6641 -> head_s3 0.6654 = +0.0013 (same trunk, ep0 gcorr head)
head -> combo_s3 0.6655 = +0.0001 (nonlinear mixture, no weight lesson)
combo_s3 -> combo_s4 0.6660 = +0.0005 (seed)
```

Complete-recipe CV (E410, seed-3 folds): pooled combo 0.693658 vs control 0.691244, **+0.002414 [+0.00156,+0.00345]**. Supplied-validation full file (~0.7005-0.7008) disagrees in ranking with public and is fitted on 1,505/1,873 sequences — not a forecast. The honest-0.7000 milestone is **unsupported by any complete predictor**; it needs +0.034 over public, ~14x the CV gain.

Incumbent (`src/wnn/nn.py:RNNModel`): linear-in → GELU → 2-layer GRU256 → head `[Linear(2H,H), GELU, Linear(H,2)]` fed by `[gru_out, projection_skip]`. Trained Muon/chunk1000/stopped-bootstrap + `gcorr` head fine-tune, equal parent/head merge. Metric is `global_wp_clip2_v1` (`METRIC.md`, `gmetric.py`): pooled clipped weighted Pearson, weight `|clip(y)|`, averaged over t0/t1.

## 2. Why the agent got stuck

Not lack of ideas — lack of transfer stage:

1. **224 activation fits + 30 sequence fits ≠ 224 chances to beat 0.6660.** Almost all are H64/H32, 1 epoch / 474 updates, 24 fit / 16 repeatedly-exposed eval components. E488 (12 continuous 3-epoch fits) is the only duration exception, still H64 on the same 16. The GRU256 + teacher/EMA + head procedure is a different predictor.
2. **Tiny-panel ranking loop.** More variants on the same 16 components add selection exposure, not public information. Component bootstrap CIs condition on fixed models; they exclude seed/search uncertainty. Three seeds ≠ fresh components.
3. **Failures misfiled as negatives** (correctly flagged in `ELEPHANTS_REVIEW_2026-10-05.md` and execution audit): W384 RAM-guard stop, E462 INT8 ABI failure, E463 TPU queued, E485 strict-gate miss, E489 import failure + 32s-late admission. None is a model negative.
4. **Narrow negatives treated as family rejections:** E468 frozen coarse codebook → "joint models dead"; hard output hold (-0.08 WP) → "events don't matter"; weak-ridge offset failing on strong GRU → "residual adapters dead"; one EMA-rows screen → "cadence dead".

## 3. Elephants (what the Oct-5 reviews underweight)

**E1. The head *is* the public gain, and its mechanism is unknown.**
Head improves WP while *worsening* WMSE on both targets/panels; prediction dispersion rises (parent std 0.98/1.05 → head 1.09/1.16 vs target 1.29/1.30, `strong_baseline/REPORT.md`). WMSE optimum is the tilted mean `E[|Y|Y]/E[|Y|]` — already implemented in E468. So the head win is dispersion/covariance optimization under a centered scale-invariant metric, not better pointwise MSE. Yet E424 parents, E421 frozen backbone, E468 frozen trunk, and all H64 pilots omit the `gcorr` head procedure. E464 (8 heads + 16 complete contrasts, now complete per ledger: "none beats seed-3 combo") actually *strengthens* the specialist null — cite it as head-inclusive, don't re-litigate parent-only.

**E2. Chunk1000/Muon/bootstrap are fully confounded.**
The +0.0014/+0.0019 public climb cannot be split. Chunk1000 doubles steps/epoch, changes EMA update counts (0.999 vs 0.999² effective), denominator statistics, TBPTT carried-state staleness. E430 (chunk accum vs chunk2000, pooled numerators), H-072 macrostep, H-073 normalizer, E414 mass-variation calibration all drafted, none fitted as complete predictor. Don't claim "longer credit won".

**E3. Seed variance dwarfs all tuning.**
Local s4-s3 +0.00187 [-0.00098,+0.00693] vs public +0.0005. Fixed s3/s4 bag: full-file 0.701107 > s4 0.700771, excluded 0.70142 < s4 0.70169 — ranking flips by slice. Two seeds is not a seed distribution. Bag unsubmitted; runtime unmeasured (likely >60min as 4-model ZIP, or distillable).

**E4. t0 vs t1 asymmetry is the cheapest structural clue.**
Excluded panel t0 0.6876 / t1 0.7120, Δ 0.024 (`VALIDATION_SCORECARD.md:11`). Same rows, same model. E426 tested *teacher-consistency* EMA clock (-0.000045/+0.00008), not input/history adapters. H-084 (t0 row-EMA / t1 event-EMA adapters), H-085/087 activity/age×EMA, H-086 i1-residual all open.

**E5. Objective mismatch is proven, unexploited.**
`weighted_pooled_corr_loss` and `PooledCorrMemory` exist in `nn.py` but incumbent trains WMSE + aux then `gcorr` head. Mechanism audit's test #2 (exact pooled-WP loss vs WMSE, same init/order, one stabilizing denominator + hybrid WP+WMSE arm) never ran at full scale. This is the only direction that directly targets the head's dispersion effect.

**E6. Width is censored, not refuted — and it's the largest single-fold signal.**
W384 f0 +0.0020 [+0.00072,+0.00332]; W320 same-recipe null exists; W384 v7 passed both T4 probes, stopped epoch0 batch120/373 at 8.59GB RSS vs 8GiB guard, host 846MB left. No checkpoint/score. Muon×width interaction unknown. QAT preserves accuracy ±6e-06 (E419) — the enabler for fitting W384 or bag into 60min via INT8, but repaired q8 has compatibility only, no runtime approval.

**E7. 0.7000 is currently a fantasy number.**
Largest complete-predictor effects: CV +0.0024, width +0.0020 (1 fold), seed noise ±0.002. Tiny-panel +0.008 (E488 mixture) / +0.030 (SSM spline) will shrink 5-10x on transfer. 0.70 needs ~10 stacked independent +0.003 wins or one structural break (memory/posterior/teacher). Do not add pilot deltas to 0.6660.

## 4. Rejected-for-nothing: keep narrow, retest bounded

| Direction | What actually happened | Still-open retest (one decisive comparison) |
|---|---|---|
| Run-relative volume (H-078) | E413 HISTORY vs CURRENT +0.000962 [-0.00008,+0.00206], 1 fold/seed (`e413_paired_report.json`) | Second fold + second seed, complete recipe + head. No longer "never tested". |
| Diagonal prefix modulation | E424 largest adapter +0.000768 [-0.00008,+0.00173], parent-only | Already head-tested in E464 (null). Only revive as *different* state allocation with direct control, not same arm. |
| Target-specific clocks | E426 EMA-consistency near-null; ridge own-vs-shared failed universally (class .6569 -0.057 vs others +) | Causal input/history adapters H-084, not teacher EMA. Frozen-strong-state screen first: both-clocks vs target-specific vs duplicated-shared, matched params. |
| Learned SSM decay | +0.00363/+0.00382/+0.00304, all 3 CIs positive — cleanest replication | Residual diagonal/LRU memory alongside GRU256, learned vs fixed decay, matched MLP head + CPU cycles. Not wholesale Mamba swap. |
| Spline readout | SSM-spline over MLP +0.038→+0.028→+0.012 (last CI crosses 0); GRU-spline +0.020/+0.018/-0.003 | Zero-init piecewise-linear residual on saved strong states vs hinge/MLP, train-only knots, then one end-to-end if positive. Tests local edges, not "KAN". |
| Attention | Positive-kernel memory mixed; SSM-attention loses 2/3 seeds | Delta-rule / erase+write event-keyed retrieval vs additive position-only control. Different recurrence, not same test. |
| Joint heads | E468 5 arms all below GRU128 parent (best -0.00055); but regression anchor also -0.00055, shared CE aux on all arms, coarse 32-atom frozen 1-pass | Train-only support + continuous zero-init residual anchored at current prediction vs WMSE-only anchor vs shuffled support, same budget, no aux CE on control. E421 (-0.088 frozen sparse) doesn't close end-to-end either. |
| HFT/event structure | Physical spread decode failed; but 3761/3763 + 3841/3842 scored jumps coincide with current i0 event; `puzzle_a` 0 changes on 97k unchanged-i0 runs | Event-conditioned latent update + quiet-run innovation path, never output freeze. Zero-init low-rank adapter on strong states: event-age/innovation vs row-age-only vs shuffled-event controls. |
| Lag/phase (puzzle_b lag4 +0.0079 → +0.0006 after removing 2 seqs; fractal row16 +0.037 vs event16 -0.055) | Period scan correctly killed | Predeclared lag/innovation feature only if beats equal-count shifted-lag sham on new components, then strong-model test. |
| Nonlinear offsets | Weak-ridge +0.022/+0.037 killed on strong GRU -0.00047 | Correct kill of *transport*; not of strong-state residual t1 corrector with component-disjoint cal (H-083). |
| Teachers TE07 0.6636 | Kills teacher-ensemble *as member* | Doesn't close H-096 crossfit / H-097 gen3 / H-098 privileged-future8 (origin-label-preserving, component-cross-fitted). |
| `state_drift_response` (stored-packet fixed lags, ≤32MiB, no GPU) | INCOMPLETE_PAUSED, cheapest discriminator | Do first. Transient vs persistent, grad vs update. |
| Warm-up/prefix (E417 reset250/500 -0.042/-0.031) | Disruption, not credit test | H-079 random origins + rows0-99 regime intercept + per-regime EMA; H-081 sentinel; H-089 cycle8-then-EMA vs row-EMA. |

**Correctly dead — do not reopen unchanged:** post-hoc calibration/scales/smoothing/soup (max bound 6e-05), EMA6 public -0.0006/-0.0010, TCN -0.0044 / LSTM -0.00086, online-norm zero-cache, merged-GRU additive (negative every weight), runtime governor, DTW/routing grids, W320-in-AdamW, hard output hold, "sum/difference rotation as model" (invertible change, no info).

## 5. New directions & priority queue

Rules for all: same generation (teacher/prop/source/epochs/seed/head) per contrast; parent vs complete reported separately; freeze components before array reads; 1 fold/seed to screen, new seed/partition to promote; report pooled WP + t0/t1 + class slices + CPU latency/ABI; no LB tuning on +0.0001 blends.

**P0 — cheap, decisive (CPU or 1-2 GPU-h):**
1. `state_drift_response` from stored packets. Falsifier: no transient/persistent separation.
2. Seal repaired-q8 runtime: loaded-host 1-CPU timing on exact repaired hash. Packaging, not science. Keep s4 37m07s anchor.
3. Bag runtime probe: measure s3/s4 4-model ZIP 1-CPU time; if >52min, bag-as-teacher distillation student (H-097) instead of upload.
4. E430 cadence: chunk1000-accum vs chunk2000, pooled numerators/denominators, fixed optimizer/EMA counts. Separates credit-length from step-count.

**P1 — transfer to GRU256 (main compute, 2 seeds each, full head procedure):**
5. Fixed 50/50 GELU-LeakyReLU readout (E488: +0.0089/+0.0043/+0.0087 at ep3, 2/3 CIs positive; recurrent-static ~0). Control: unchanged GELU + frozen-swap arm. One change only — not + embedding/candidate.
6. Bounded recurrent candidate (E478 H64 α0.1: +0.0027/+0.0107/+0.0045, 2/3 CIs positive) with slope-matched linear/cubic control (E473/E481 warn sine≈slope). One placement, full schedule.
7. Pooled-WP objective (mechanism audit #2): WMSE vs exact pooled WP vs hybrid, same init/order. Watch grad norms/clipping incidence.
8. W384 fold1 + Muon×width, after honest host-memory fix (stream/chunk buffers, Arrow/pin accounting) + CPU benchmark. QAT/INT8 only as deployment enabler.
9. Volume28 second fold/seed (H-078); distilled normalizer aligned/shuffled/historical (H-073); properly normalized accumulation (E430).

**P2 — representation (frozen-strong-state screen → one end-to-end):**
10. Event-aware posterior (top structural bet): event-age + innovation summary, quiet updates allowed. Controls: param-matched row-age adapter, shuffled-event diagnostic (nondeployable, diagnostic only).
11. Target-asymmetric adapters H-084/H-085/H-087 + i1-residual H-086 vs i0-lag/shifted-i1 sham.
12. Learned diagonal/LRU memory + compact spline/hinge readout as residual branch; learned vs fixed decay; spline vs matched MLP; train-only knots; CPU cycles before enlarging state.
13. Joint support + continuous escape residual anchored at current pred (addresses E468 limits + difference-mode residual without assuming mechanism).

**P3 — teachers/privileged (reserve structural slice, most expensive):**
14. H-096/097 crossfit + gen3; H-098 future8 privileged teacher predicting *origin* label, component-cross-fitted, origin weights preserved. Test teacher value before multigen program. Falsifier: no student transfer / leakage / oracle-at-inference dependence.

## 6. What not to do

- No more H64/H32 sweeps on the same 16 components. Next budget majority to complete-predictor comparisons + confirmation.
- No averaging per-class WP to get official score; no full-file 0.701 → public 0.701 mapping; no summing pilot deltas onto 0.6660.
- No retrospective tolerance widening (E485 strict gate stands) or converting deployment checks into predictive conclusions.
- Preserve every arm, every failure, lineage/overlap lists. Operational failures stay in engineering column, never scientific-negative column.

**Strongest underused idea:** a model of evolving conditional beliefs — event-conditioned state + quiet-run innovations + targeted memory + target-asymmetric readouts, evaluated against the preserved strong predictor with the real head procedure. Evidence motivates it; nothing yet proves it.

---

