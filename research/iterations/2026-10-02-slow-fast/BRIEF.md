# Wunder Fund "Alpha Connectome" — full research brief (state as of 2026-10-03)

This archive is for an independent research agent. Goal: generate ideas that could lift our **public LB from 0.6642 to above the leader (~0.6677)**. Read this file first; everything else is supporting material.

## 1. Competition

- Host: Wunder Fund, https://wundernn.io/connectome (docs are a JS app; copies in `docs/`). Deadline 2026-11-15. 5 submissions/day.
- Task: streaming prediction of two anonymized targets `t0`, `t1` ("future price movement indicators" of instrument `i0`), row by row.
- Data: `train.parquet` 10,607 sequences, `valid.parquet` 1,873 sequences (with true scoring mask `is_scored`), each sequence exactly 20,000 rows. Test: 1,970 sequences (hidden), fed row by row.
- Row = 112 float32 features: for two instruments i0, i1: 11 bid price-like + 11 ask price-like + 11 bid volume-like + 11 ask volume-like + 4 trade price + 4 trade volume (52 each), plus 8 additional features a0..a7. Features are anonymized: rank-transformed/clipped (mass at ±2.32); column index is an identifier, NOT a depth level; price and volume columns are full-rank mixtures of the underlying book.
- Rows 0–98 are warm-up (no prediction). State must reset when `seq_ix` changes; no cross-sequence information; strictly causal.
- **Metric (verified on the LB itself):** pooled |y|-weighted Pearson over all scored rows of all sequences (not per-sequence), targets and predictions clipped to ±2, weight = |clip(y)|, mean over t0 and t1. The scored mask (~9.4% of rows, sticky runs) is hidden at test. NOTE: `docs/data_overview.md` is the stale starter-pack text that describes per-sequence averaging; `METRIC.md` is current. Evidence: scaling class-3 predictions ×0.7 changed public by −0.0083, matching the pooled formula (−0.009 locally); per-sequence scoring would be invariant to that.
- Runtime: 1 vCPU, 16 GB RAM, no GPU, 60 min for the whole test (a competitor saw a timeout at ~70 min). Deterministic output required. Our packages run a native C++ GRU engine (int8 dynamic quant / fp16) with Python wrapper.

## 2. Leaderboard
- Leader ~0.6677 (user report 2026-09-29). We are 19th at 0.6642. Public notes of two competitors at 0.6569 (#11) and 0.6581 exist (see `scout/`). Nothing public from the leaders.

### Our public history (all scores public LB)
| package | members / change | public |
|---|---|---|
| final_submission (09-21) | early | 0.6516 |
| H41 (09-25) | | 0.6519 |
| (09-27) GRU pair since/gclp | | 0.6622 |
| candB (09-29) | h256 + 2×h128, head FT | 0.6639 |
| candA | 2×h256 (governor dropped a member) | 0.6629 |
| candC | distilled d256 + 2×h128 + c500 | 0.6640 |
| **V1** candV1_ft2 | candC with 2-epoch head FT ("H2") | **0.6641** |
| V2 | parents with valid ×3 | 0.6611 |
| V3 | parents train-only (+1-ep head FT) | 0.6629 |
| midpoint student | 1-layer 256 student + 3×h128, int8 | 0.6639 |
| e6a | V1 members + 2×1L256 (6 members) | 0.6641 |
| p_c3zero | midpoint with class-3 outputs zeroed | 0.5473 |
| p_c3only | midpoint with only class-3 outputs | 0.4017 |
| p_c3x070 | midpoint with class-3 outputs ×0.7 | 0.6556 |
| v1par | V1 members at parent epoch (no valid head FT) | 0.6630 |
| v3par | train-only parents, no head FT | 0.6606 |
| v1_dm | V1 recipe with a-feature demean on all members | 0.6639 |
| v1_dm5 | v1_dm + second d256 | 0.6641 |
| **final_mix5** | raw d256 ×2 + demean h128 ×3 | **0.6642** (best) |

## 3. Data / label facts we established
- 4 market classes by target quantum (tick size). Class 3 = large tick (quantum 0.65687), ~8 book states, ~37–43% of sequences, ~43–46% of metric weight, worst class (within-class WP ~0.45 vs 0.73–0.81 for classes 0–2). Class mix differs a lot between train and valid.
- **Label law (class 3, exact on near rows):** F = round(mean mid over the next 4 i0 book-price events). With bid/ask/spread sp in ticks: inside the spread t0 = F − ask, t1 = bid − F; if F above ask, t1 = −(sp+2); below bid, t0 = −(sp+2). Targets are PnL of "buy at ask"/"sell at bid" marked to F with a capped losing side. t0+t1 = −sp only inside; otherwise encodes unsigned move magnitude. t0−t1 = 2(F−mid) = direction. Oracle direction → WP 0.976.
- Targets change ONLY on rows where the i0 price part changes (100%): labels are event-indexed. On far-horizon rows (4th next event ≥64 rows away) the 4-event law matches only 82%; clock-based definitions match ≤56%; truncated-K variants do not explain the rest. 18% of far labels depend on something not visible in the price codes (unresolved).
- i0 price codes for class 3 decode to tick paths via a learned transition table (cache `class3_moves_full.pkl`); rebuilt labels are 90% exact overall but the failures carry 25% of metric weight (concentrated on far-horizon rows).
- In class-3 sequences the second instrument i1 uses the same code table (82% of transitions known) and is itself a quiet large-tick instrument (price events in 6.5% of rows vs 38.5% for i0).
- a-features: a6 ≈ real time since last i0 book change, a5 ≈ time since last trade, a2/a3 book-event intensities (E-091); a2/a3/a4 predict the scoring mask.
- Scoring mask is predictable from features (AUC ~0.93); scored rows come in sticky runs.

## 4. Where we lose on the hidden test (public probes)
| | local valid (OOF) | public | gap |
|---|---|---|---|
| full | 0.6906 | 0.6639 | −0.027 |
| classes 0–2 only (c3zero) | 0.5522 | 0.5473 | −0.005 |
| class 3 only (c3only) | 0.4260 | 0.4017 | −0.024 |
Almost all of the valid→test loss is class 3. Class-3 output scale is already optimal on test (×0.7 hurts). Within class 3, per-sequence WP falls 0.14 per unit of a "quiet" index (fewer distinct volumes, lower trade-price std, higher a5/a3): 0.61 active vs 0.40 quiet. Valid class 3 is quieter than train; test looks quieter still. In quiet sequences scored rows sit ~25 rows before the next target change (vs ~9 active): longer horizon. All 28 cached model variants degrade identically on the quiet tail.

## 5. Model / pipeline (current best recipe)
- GRU (2×128, 2×256, or 1×256) on raw 112 features + engineered event features (time since events etc.), TBPTT chunks of 2000 rows (500 for the "c500" member), batch 32, AdamW lr 1e-3 OneCycle (5% warmup), wd 0.01, 4 epochs, EMA teacher (decay 0.999, evaluate EMA weights), fp16.
- Loss: weighted MSE with weight |y| × propensity(scored)^2 + 0.03 (propensity model trained on valid masks); valid rows use their true mask (eps 0.03); aux heads (aux-struct 0.3, aux-next 0.3); since-weight.
- Distillation: d256 student = 2×256 trained toward 0.5·y + 0.5·mean(teachers h256 seed3/seed4).
- Data: parents trained on train + all valid (×1); then "H2" = head-only fine-tune (trunk frozen) on valid only with the true scoring mask and a pooled-correlation loss (gcorr), 2 epochs, lr 5e-4.
- Ensemble: 4–5 members averaged; native C++ engine, int8.
- Local evaluation: chain-grouped valid folds (sequences are pieces of longer time streams; chains reconstructed), OOF on folds 0/1, 936 sequences; pooled WP; plus a "quiet class-3 tail" slice (121 seqs) as a test-shift proxy. Seed noise ≈ 0.0005–0.0009 on all-WP.

## 6. What moved PUBLIC (the only trustworthy signal)
- Valid in parent training ×1: +0.0024 (train-only no-FT 0.6606 → 0.6630). ×3: −0.003.
- Valid head FT with true mask: +0.0011 (0.6630 → 0.6641); +0.0023 on train-only parents.
- EMA weights (earlier): +0.0017. Distillation/width: ~+0.001 historically.
- Ensemble width beyond 4: 0. demean of a-features: 0 (−0.0002). Class-3 scaling: harmful.

## 7. Refuted / null (do not re-propose without a new mechanism)
Local unless noted; numbers are deltas vs matched control.
- Capacity: h384, 4.9M causal transformer (= h256), minGRU, event-token attention, dual-clock and event-only GRU (quiet gap unchanged), two-stream i0/i1, class-conditional heads, class-3 expert, quiet-regime expert (c3quiet 0.26 vs 0.40).
- Losses/targets: atom (distributional) head, rational 3-branch readout (gain was oracle-spread leak), far-horizon aux targets (aux-next 1.0, aux-next-k 2,4: quiet −0.001/−0.005), gcorr from scratch, V-REx.
- Data: valid ×3 (public −0.003), quiet-regime reweighting (capacity trade only), i1 role-swap augmentation with rebuilt labels (−0.004 c3), + domain flag (same), duplicating class-3 seqs with rebuilt labels (−0.008 c3 vs true labels: label noise is the killer), noisy student on i1 (teacher WP on i1 only 0.21), bid/ask block swap is not a mirror symmetry (true mirror needs code decoding — untested).
- Calibration/post-processing: per-class/per-regime scale and offset (≤ +0.0004 oracle on valid; ×0.7 public −0.0083), online per-sequence recalibration from rebuilt matured labels (E-080: 0.672 → 0.647), sequence-level level: oracle would add +0.031 but residual level is not persistent within a sequence (corr 0–0.1), disagreement-based shrinkage (members corr 0.98–0.99).
- Training: dropout 0.1/0.2 (0), wd 0.1 with EMA (0), 6–12 epochs (EMA plateau, h256 overfits after epoch 3), SWA (0), 4-teacher distillation (+0.0008, noise), head-FT variants (4 epochs, full-FT, low-lr full FT: ±0.0005), gcorr head FT on train+valid (−0.007: the FT gain needs VALID's TRUE MASK), valid FT without mask weighting (gain disappears), k-fold valid bagging (nested simulation: no gain beyond plain seed ensembling).
- Features: 114-feature heuristic library (E-089: 0), GBM residuals (overfit), lagged i1 flow, order-flow sums, decoded queue factors (ICA, AUC 0.62 like raw), state resets (no position decay in our models), a-feature demean (helps h128 locally +0.003 quiet, hurts distilled d256, public 0).

## 8. Running / untested at time of writing
- Muon optimizer (hybrid Muon on GRU matrices + AdamW), fine HP sweep (lr 2e-3/5e-4, batch 64, warmup 20%, β2 0.99, clip 0.3) — results pending.
- True bid/ask mirror augmentation via code decoding (graph isomorphism of 119 class-3 book states / ICA pairing of volume codes).
- Better scoring-mask model for train weighting (the FT result suggests "train exactly on scored rows" is what transfers).
- Resolving the far-horizon part of the label law (real-time windows?).

## 9. Key open questions for idea generation
1. What else transfers to the test like valid-with-true-mask does? (Is there a way to make train rows "look scored" more faithfully?)
2. Class 3 quiet regime: information-limited for causal models, or are we missing an observable (real time from a5/a6, i1 dynamics, queue depletion)?
3. Any representation change that makes predictions of the direction t0−t1 better on far-horizon rows?
4. Ensembles of our GRUs are near-copies (corr 0.98–0.99): what genuinely different model would have independent errors?

## 10. Archive map
- `BRIEF.md` (this file), `docs/` official docs copy + `METRIC.md`, `scout/` competitor/literature scouting summaries.
- `research/` — project ledgers: `CURRENT_STATE.md`, `frontier.md`, `game_changers.md`, `tree.md`, `experiments.jsonl` (E-ids), `hypotheses.jsonl`, dated findings, `lit/`, `ideas/`, `iterations/*` (readouts, small JSON, analysis scripts).
- `code/` — `scripts/train_rnn.py` (trainer with all flags), `predict_run.py`, `fold_job.sh`, `fold_eval.py`, `i1swap_aug.py`; `src/wnn/*` (features, ticks/label law, metric); `submission/solution_dm.py` (inference wrapper); `native_cpp/` (C++ engine sources).
