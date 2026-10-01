# Brief for architecture / learning / optimization scouts (2026-09-28)

## Task
Streaming prediction. Each sequence = 20,000 rows × 112 float features (order books of two
instruments + trades + 8 anonymous clocks). Predict two targets t0, t1 (future price movement of
instrument i0) at rows 99..19999, row by row, causally, state reset per sequence. Score = mean over
the two targets of the **global pooled weighted Pearson** over rows chosen by a hidden mask
(~9% of rows), targets and predictions clipped to [-2, 2], weights |y|. Test is an unseen period.
Deployment: 1 vCPU, 60 min for ~39.4M rows (≈90 µs/row total), ZIP ≤ 20 MB, Python 3.11; we run a
native Rust GRU engine (h128 × 2 layers ≈ 18 µs/row per member on the scorer; 3–4 members fit).

## Current recipe (public best 0.6622, leader 0.6677)
GRU h128 × 2 layers, linear input projection of 211 inputs (112 raw + 99 causal event features),
MLP head; loss = |y|-weighted MSE with propensity² row weights (floor ε 0.03) from a model of the
scoring mask, true masks on validation rows, aux heads (next target change, spread/branch/class);
EMA mean-teacher (decay 0.999, 4–6 epochs, one-cycle LR 1e-3, AdamW wd 0.01, batch 32 × 2000-row
TBPTT chunks); stretch-age ("since") row weights; then two-stage LP-FT on validation rows with the
true mask (head only, then all, 1 epoch each, pooled-Pearson loss). Final = equal blend of 2–4 such
GRUs (+ optionally the train-only organizer GRU baseline ×0.25).

## What moved the public score (and did not)
- Worked: output scale vs clip (+0.057); |y|-wMSE; propensity weights/floor; true masks; valid
  LP-FT (+0.0014); EMA mean-teacher (+0.0017); blending with the organizer baseline (+0.0021);
  two-seed pair (+0.0004).
- Flat / negative (≥ 40 experiments, see research/experiments.jsonl): width 192, depth 1/3, LSTM,
  conv / token / MLP encoders at EMA-4, dropout 0.1, wd 0.1 on EMA-6 (E-256), 8+ epochs without EMA
  (memorizes label noise), plain CE / Huber losses, distillation α 0.5, TD targets, many auxiliary
  heads (tier, sign, mask, quote path), online per-sequence recalibration, feature additions,
  post-hoc calibration, per-class blend weights.
- Regime: label-noise dominated; more data helps (+0.002–0.003 per doubling), more epochs over the
  same data memorize; EMA / averaging is the only thing that let longer training help.
- Ensembles: members make near-identical errors (residual corr ≥ 0.99); structurally different
  members (chunk-500 TBPTT, online-normalized inputs, the train-only baseline) add most.
- Validation proxies are fragile: a proxy calibrated on 10 public scores (E-261) prefers softened
  masks and train-unlike sequences; the train-only baseline is undervalued by raw validation proxies.
- Market class (4 instruments) is known causally at row 99 from the model's aux head (100%).

## Rules for scouts
- Read-mostly. Do not edit existing repo files; write only your own report file named in your task.
- Resource caps: prefix any python with `WNN_MEM_MAX=6G WNN_CPU_PCT=20 scripts/limited .venv/bin/python`;
  no GPU; never `pkill -f`; one heavy process at a time; no pyarrow memory_map.
- Competition rules: no external data in the model, no reading current Wunder-competition
  participants' public repos, no leaderboard probing.
- Output: hypothesis cards, each with mechanism, source (paper/book/page or URL), exact change to
  `scripts/train_rnn.py` (or native engine) with flag names, runtime cost at inference, cheapest
  fold-0 test (our fold runs take ~25–40 min of GPU), predicted effect, falsifier, and overlap with
  failed experiments above. Rank by expected public gain per GPU-hour.
