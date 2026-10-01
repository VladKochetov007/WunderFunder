# Creative round context (2026-10-01 evening) — read this, not the full history

## Contract
- objective_mode: **competition** — maximise expected *unseen* public Global WP under serving rules. Discovery of a
  new mechanism is valued only insofar as it can move the deliverable.
- Question: **where to dig** — find a representation change / factorisation / reverse-engineering insight that can
  plausibly move public score by ≥ +0.002 (ours 0.6622, team 0.6641, leader 0.6677), not another +0.0003 tweak.
- Verification tier: B/C. All local evaluation data is **exposed** (4 clean chain-excluded folds and the 368-seq holdout
  have been inspected many times). Fresh confirmation exists only as the public leaderboard (user uploads).
- Budget per agent: CPU only, ≤ 3 cores, ≤ 8 GB RAM (`WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited <cmd>`), no GPU
  (controller owns it). Write only under your own `research/campaign_2026-10-01/creative/<agent>/` dir and new
  `scripts/cr_<agent>_*.py`. Never edit shared code, .venv, native/, submission/. Never git reset/stash/commit.
- Clean room: do not read source code on git branches origin/quant-probes, origin/cpp-engine,
  origin/research/creative-iteration-20260930, origin/archive/2026-10-01 or the wunder/ dir. Teammate ideas/numbers
  in `research/campaign_2026-10-01/TEAM_INTAKE.md` are fine.
- Data: competition data only for any fitted mapping (`datasets/train.parquet`, `datasets/valid.parquet`); fit on TRAIN
  or on folds disjoint from what you evaluate; never use targets/is_scored as inference inputs.

## Task
Predict t0, t1 (future price-move indicators of instrument i0) per row from causal streams: 20,000 rows/sequence,
112 anonymised features (i0/i1 22-level price-like and volume-like codes, 4 trade price/vol, a0..a7), rows 0–98
warm-up, state resets per sequence, CPU 1 vCPU, 60 min for ~39.4M rows, one Python callback per row.
Metric: **Global Weighted Pearson** — pool all rows with need_prediction ∧ is_scored across all sequences, clip y and
p to ±2, weight |y|, one correlation per target, mean of the two. Scoring helpers: `scripts/screen_score.py`
(frozen math), `scripts/de_public_proxy.py` (public-like class mix), `src/wnn/gmetric.py`.

## Hard facts established (TRAIN/validated)
- Targets are **piecewise constant** between quote-move events; they change exactly on rows where `a6` hits its lower
  endpoint (100.000% of |y| weight, no unknown mass). ~37% of class-3 rows have y = 0 (zero metric weight).
- Market class (tick quantum of targets; 4 classes) is detectable causally from warm-up with 100% accuracy.
  Class 3 = 37% of valid sequences, ~47% of metric weight, local WP only 0.43 (classes 0–2: 0.81/0.78/0.74).
- Local↔public gap is **class composition**: public ≈ 46% class-3 sequences (local 37%). Within-class WP does not
  depend on chain length.
- Interior t0+t1 = −spread(current price code) exactly; ~20% of metric-weighted variance is fixed by current state,
  ~80% is the future direction D = (t0 − t1)/2. Structural spread heads lose.
- For weights |y|, the WP-optimal predictor is affine in E[|y|y|x]/E[|y||x] = minimiser of |y|-weighted MSE — the
  parent objective is aligned; per-class scale is already calibrated.
- Upper bounds (oracle, not achievable): per-sequence level of D → 0.727, per-sequence scale → 0.719 vs ~0.70 local.

## Current best models
- Since-weighted GRU128 ×2 (wMSE + aux struct/next heads, propensity-weighted rows, EMA, epoch 3) + head-only gcorr
  fine-tune. New tonight: averaging parent head and FT head on one trunk helps class 3; routing class 3 → that mix,
  classes 0–2 → two-stage FT pair (RSP2): holdout +0.00125, folds 2/3 +0.0003. Distilled GRU256 adds ~+0.0005 locally.
- Cached out-of-fold predictions (clean folds 0–3) on valid: `cache/pred_c20260929_ft_since_s{3,4}_f{k}_valid.npy`
  (two-stage FT), `..._f{k}h` (head FT), parents `cache/pred_c20260929_since_s{3,4}_f{k}_valid.npy`; rows ordered as
  `cache/valid_targets.npz["row_group"]` (fields y, mask, row_group, market_class). Fold membership:
  `research/campaign_2026-09-29/artifacts/splits/fold{k}.json` via `wnn.splits.load_campaign_split`
  (`evaluation_groups`, `components`).

## Already tried and rejected (do not re-propose without a genuinely new angle)
Width h192/h384, LSTM/extra depth, static MLP, transformer (XL, event-attention), two-stream/dual-clock GRUs,
periodic reset, class-3 experts/atom heads/context tables, quant order-flow features, decoded tick features,
level aug/drop, V-REx, privileged teachers, gen-2 distillation, matured labels (leak), online recalibration,
post-hoc calibration, per-class scale/weights, cross-fitted blend weights, joint ensemble head FT, pooled-memory
correlation loss, spread-label aux, structural S/D head, output smoothing/sharpening, mirror augmentation
(not identifiable), extra GRU seeds/head bags, chunk-500/online-norm diversity trunks, inverse-clock time warp,
event-content memory (no WP gain), bilinear cross-member head.
