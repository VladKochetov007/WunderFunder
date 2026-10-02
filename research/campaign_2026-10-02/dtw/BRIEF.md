# DTW / piecewise-constant idea round — brief (2026-10-02 04:15, deadline for reports 06:45)

User question: "Use DTW. Propose different ways to apply it. Our target is piecewise constant, but predictions are
not. What can we do, how to model?" Read `research/campaign_2026-10-01/creative/CONTEXT.md` first (task, metric,
hard facts, rejected list). Objective mode: competition — maximise expected unseen public Global WP.

## New facts since CONTEXT.md
- Public (2026-10-02): distilled GRU256 student alone (alpha 0.5) = **0.6641**; + 2x GRU128 since members = 0.6633;
  class-routed version 0.6638. Team best 0.6641, leader 0.6677. Distillation alpha 0.7 vs 0.5: single student
  +0.0023 on fold 0 (fold-1 confirmation running). Stronger/smoother targets help; extra members dilute.
- E-323 residual analysis: half of lost metric sits in class-3 runs (segments between target changes) lasting
  64–1,024 rows; corr(prediction, run target) at the run's first row is 0.02 in class 3. Member residuals
  correlate 0.985–0.999. alpha-0.7 gain sits in rows >= 32 rows before the next quote move.
- E-295: causal per-sequence EMA smoothing of predictions (across segments, no reset) LOSES (alpha 0.3: -0.001).
  Segment-reset / hold variants were NOT tested.
- External audit notes (treat as hypotheses): long quiet intervals may need a fast within-interval state plus a
  slow event-clock state; "restricted probe found nothing" is not an information ceiling; outcome-dependent
  distillation weights (H-059: alpha grows with rows-until-label-change) are not automatically unbiased.

## Piecewise-constant algebra (verify, then use)
Within a segment s (rows between consecutive target changes; boundaries are causally observable: the target changes
exactly on rows where a6 hits its lower endpoint) y is constant and |y| weight is constant. Pooled weighted
covariance then only sees the length-weighted segment mean of p; within-segment variation of p adds to Var(p)
without adding covariance *in hindsight*. Causally, within-segment updates are still legitimate when information
arrives (elapsed time, trades, i1 moves). The question is how much within-segment variation is information vs noise.

## Data / tools
- `datasets/train.parquet` (212M rows, 117 cols: seq_ix, step_in_seq, need_prediction, i0_*/i1_* codes, trade cols,
  a0..a7, t0, t1). NEVER load it whole: read selected columns and <= 300 sequences (pyarrow filters / row groups).
- `datasets/valid.parquet` (1,873 sequences x 20,000 rows).
- `cache/valid_targets.npz`: y (1873,20000,2) f32, mask (scored rows), seq_ix, row_group, market_class.
- Out-of-fold predictions of the current best single model family, rows aligned with valid_targets:
  parent `cache/pred_c20261001_d256_s3_f{k}_valid.npy`, head-FT `cache/pred_c20261001_ft_d256_s3_f{k}h_valid.npy`
  (k = 0..3; only fold k's evaluation sequences are out-of-fold in file k). Fold membership:
  `wnn.splits.load_campaign_split(Path('research/campaign_2026-09-29/artifacts/splits/fold{k}.json'), ROOT)
  .evaluation_groups` (row_group ids -> rows of valid_targets via its row_group array). Pool the 4 folds' OOF rows
  to get one OOF panel. Metric: global weighted Pearson (clip y,p to ±2, weight |y|, per target, mean of t0,t1);
  reuse `src/wnn/gmetric.py` or the math in `scripts/screen_score.py`. Report deltas vs the unmodified OOF panel,
  per class and pooled, plus a public-like class mix [0.183, 0.160, 0.198, 0.459].
- Prior EDA code you may reuse: `research/campaign_2026-10-01/eda_cells/*.py` (run anatomy, error budget).

## Contract
- CPU only. Every compute command via `WNN_CPU_PCT=12 WNN_MEM_MAX=3500M WNN_SLICE=wnn-campaign.slice
  scripts/limited <cmd>` (2 cores, 3.5 GB). Memory-map npy files. A GPU training job runs on this machine; never
  touch the GPU, systemd units, queues, or other agents' dirs. OOM = failure.
- Write only under `research/campaign_2026-10-02/dtw/<your agent dir>/` and new `scripts/cr_dtw_<agent>_*.py`.
  Never edit shared code, .venv, native/, submission/, ledgers. No git commands that change state.
- Clean room: do not read origin/quant-probes, origin/cpp-engine, origin/research/creative-iteration-20260930,
  origin/archive/2026-10-01 or the wunder/ dir.
- Causality: any inference-time rule may use only rows <= t of the same sequence (+ fixed constants fitted on TRAIN
  or on folds disjoint from the evaluated rows). Targets/is_scored are never inference inputs. Training-time targets
  may use future labels (that is what labels are) — say so explicitly when you do.
- Serving budget: 1 vCPU, ~60 min for ~39.4M rows incl. a GRU256 (~36 min used). A new inference-time component
  must cost <~ 2 microseconds/row amortised, or run only on event rows.

## Deliverable (`research/campaign_2026-10-02/dtw/<agent>/REPORT.md`)
1. 3–6 hypothesis cards: representation change, causal mechanism, predicted observable + effect direction,
   cheapest discriminating test, falsifier, cost, mechanism family. At least two must use DTW (or a DTW variant:
   soft-DTW, LB_Keogh, DBA, DTW-kNN, shapelets, constrained warping) in a non-cosmetic way; others may answer the
   piecewise-constant question by any means.
2. Run the cheapest probes for your top 1–2 cards on the OOF panel / TRAIN subsample and report numbers
   (pooled + per class + public-mix, fold-consistency). Negative results are valuable — report them plainly.
3. A ranked recommendation: which (if any) deserves a ~60 min GPU run, with the exact change to make.
Keep the report under ~150 lines; put scripts and raw outputs next to it.
