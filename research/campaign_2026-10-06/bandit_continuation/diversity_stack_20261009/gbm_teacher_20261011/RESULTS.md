# H-347 GBM teacher-family test (fold 1), 2026-10-11

Verdict: NO-GO for GPU distillation from a GBM family. Cost so far: about 45 CPU-min on 1-2 cores, no GPU.

## What was run (all independently written here; limits 1 core / 5 GB after coordinator notices)
- `gbm_features.py`: causal features per sequence, 865 raw candidates, 579 after greedy corr>=0.9 pruning (`kept_features.txt`, `kept_idx.npy`, pruning on a 100-sequence pilot, `prune.py`).
  Blocks: 112 raw + 99 event features (src/wnn/features.event_features) = the 211 model inputs; lag diffs (1/3/13), rolling mean/std/EMA/rank/deviation (3/13/21) of 29 group series (mean bid/ask px, mid, spread, volume imbalance, bid/ask vol, trade px/vol, px range per instrument, mid difference, a0..a7); event-flag counts and px/vol diff sums over 3/13/21; cross-sectional skew/kurtosis per column group and row-wise corr (i0 vs i1 vol/px, bid vs ask vol), raw and rolled over 3/21. Column indices are anonymous, so "top-of-book" is replaced by group aggregates (documented deviation).
- Causality check: features on truncated sequence (5000 and 12345 rows) are bit-identical (max abs diff 0.0) to the full-sequence prefix.
- Training: 300 random fold-1 `fit.train` sequences (seed 0), rows >=99, importance-sampled (p=0.03, +0.10 if any |clip y|>1; weights |clip(y,2)|/p), 389,698 rows. LightGBM 4.7.0 (not installed in .venv; installed to a scratch --target dir, appended to sys.path), 31 leaves, lr 0.08, min_leaf 400, ff 0.5, bagging 0.7, L2 10, 1 thread, early stopping on 10% component-disjoint train holdout (best_iter 102 / 110). No eval-fold data used for fitting.
- Models: `lgb_t0.txt`, `lgb_t1.txt`; reports `fit_report.json`, logs.

## Results (fold-1 eval, official global_wp_clip2_v1)
Full 457-sequence prediction was not run (about 40 min at 1 core); a random 120-sequence subset (seed 1; `eval_groups_subset120.json`) was scored. Note the subset is easier than the full fold: student WP 0.6985 there vs 0.669 full.
| | WP | t0 | t1 |
|---|---|---|---|
| GBM (subset120) | 0.6177 | 0.631 | 0.604 |
| student h230_f1_s3_ep3 (subset120) | 0.6985 | 0.714 | 0.683 |
So GBM is 0.081 below the student on identical sequences (about 88%). Against the GRU scratch teacher (0.666 full fold) the comparable GBM level is probably ~0.58-0.60 (inference from the same ratio, not measured).

Blend (1-w)*student + w*GBM, w grid 0..0.3, weight chosen on one component-disjoint half and applied to the other (cross-fit), paired component bootstrap (1000, 120 sequences): cross-fit delta -0.0003 (95% CI -0.0008..+0.00005); fixed w=0.1: -0.0011 (-0.0024..+0.0002); w=0.2: -0.0034 (-0.0061..-0.0009). Std-matched GBM scaling gives the same sign (-0.0004, -0.0013, -0.0042). Selected w was 0 or 0.05. (`eval_blend_subset120.json`, full table inside.) Consistent with E-093.

## Interpretation and decision
- (a) OOF-level WP is only borderline (0.618 on an easy subset, ~0.58-0.60 projected to the full fold vs the 0.60 gate and the 0.666 teacher): fails the "near GRU teacher" bar.
- (b) No complementary signal: blends do not improve the student at any weight, so the errors are not diverse in a useful way at this strength.
- Step 4 (2-way OOF predictions over the 10.5k fold-1 train sequences) was not produced: at about 3-5 s per sequence this is >10 CPU-hours at 1 core, and the gate (a) and (b) evidence does not justify it. Caveats: small 300-sequence fit, only 100-110 rounds, no tuning, 1 seed; a larger fit might gain a few points but the blend headroom from E-093 (same conclusion with 3000 seqs) argues against it. Reopen only on a new representation (e.g. GBM on GRU hidden states or residual-to-student target).
