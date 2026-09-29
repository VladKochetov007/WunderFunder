# Quant-probe campaign (branch `quant-probes`, beleriand), 2026-09-29

## Question
Do order-flow features from market microstructure, or a periodic GRU state reset (a competitor's
public trick), improve the reference single model on held-out, time-isolated validation chains?

## Reference and protocol
- **Reference recipe** (`scripts/fold_job.sh`): the members of the public 0.6622 pair without the
  fine-tune stage. GRU 128x2, train + other-fold valid, wMSE, aux-struct 0.3, aux-next 0.3,
  propensity^2 weights (fold-specific `prop_f{k}`, trained without the held-out fold) + eps 0.03,
  true valid masks, EMA-4 mean teacher, since-weight tau 32 floor 0.25, seed 3.
- **Folds**: `scripts/build_chain_folds.py` rebuilds `cache/chain_folds.npz` (it was not in the repo):
  4 chain-grouped, class-balanced folds. `pure` marks validation sequences whose chain has no train
  member (no time-adjacent training neighbour); ~420 per fold.
- **Score** (`scripts/fold_eval.py`): exact pooled clipped |y|-weighted Pearson on need_prediction AND
  is_scored rows of pure fold sequences; paired delta vs reference; chain bootstrap SD.
- **Gate**: delta >= +0.0005 on the two folds pooled, same sign on each fold, P(delta <= 0) < 0.1.
  Single seed, so a pass is provisional until a second seed replicates it.

## Data checks
- Starter pack re-downloaded (sizes exact); every row group of train and valid decodes; row group
  528 (corrupt on the old host) is fine here.
- Chains reproduce exactly: 9,191 links, 3,288 chains, 90 train/valid mixed links.
- Target law (`scripts/check_target_law.py`, 400 valid sequences, scored rows): t0 + t1 changes only
  on i0 price-code events, but it is **not** a function of the current price code on high-weight rows
  (exact for 51% of class-3 weight, 13% of class-0 weight), and it takes zero/negative "spread" values.
  Consistent with two stop-limited trades (long at ask, short at bid): the losing side is floored, so
  the half-sum carries future information on large moves. The half-sum holds 5-18% of weighted
  target variance by class; the half-difference (future move) holds the rest.

## Results
(filled in below as runs finish)

### Reference (seed 3), all fold sequences / pure sequences
| fold | all sequences | pure pooled (both folds) |
|---|---:|---:|
| 0 | 0.7005 | 0.6989 over 860 pure sequences of folds 0+1 |
| 1 | 0.6765 | |

### R1: periodic state reset (competitor's `--state-reset`) — refuted
Reset every N rows in training and prediction (chunk 2000). Pure chains, folds 0+1, 860 sequences.

| variant | delta vs ref | boot SD | fold 0 | fold 1 | class 3 |
|---|---:|---:|---:|---:|---:|
| reset 2000 | -0.0007 | 0.0007 | -0.0002 | -0.0011 | -0.0019 |
| reset 4000 | -0.0009 | 0.0006 | -0.0007 | -0.0011 | -0.0031 |

Why it does not transfer: the reference WP is flat along the sequence (rows 99-1000: 0.699,
5k-10k: 0.688, 15k-20k: 0.702), so there is no late-sequence state degradation to fix; the
competitor's model lacked the since-event counters and event features that carry long context here.
Artifacts: `research/artifacts/quant_probes/rst*_vs_ref.json`.

### R2: order-flow sums over the quiet interval (`--quant-feats`, 31 inputs) — no gain
i0 volume codes minus their value at the last i0 price event (22), sums of trade-volume and
trade-price codes over trade rows since that event (8), log trade count. Batch/streaming parity exact.

| comparison | delta | boot SD | fold 0 | fold 1 |
|---|---:|---:|---:|---:|
| quant vs ref, all classes | -0.0002 | 0.0006 | -0.0002 | -0.0002 |
| within class 3 | +0.0008 | | | |
| classes 0/1/2 | -0.0007 / -0.0009 / -0.0002 | | | |
| class-3-routed (quant on class 3, ref elsewhere) vs ref | +0.0002 | 0.0005 | -0.0002 | +0.0005 |

Reading: the GRU already integrates this flow from per-row deltas; explicit window sums add
nothing measurable. The class-3 sign is the only hint, far inside noise. Routing tool:
`scripts/route_eval.py`.

### Seed noise dominates single-seed deltas (methodology)
The same reference recipe with seed 4 vs seed 3: **+0.0009**, chain-bootstrap SD 0.0005,
P(delta <= 0) = 0.04 (fold 0 +0.0002, fold 1 +0.0014; class 3 +0.0020). A pure seed change passes
the old +0.0005 single-seed gate. The chain bootstrap measures sampling noise over sequences, not
optimisation noise, so single-seed deltas below ~0.001 are not evidence either way.
New protocol: compare **seed-pair averages** (the deployed ensemble form):
`fold_eval.py "cand+cand_s4" "ref+refs4"`. Reference pair 0.7000 vs singles 0.6989 / 0.6997.

Re-read with this in mind:
- reset: the two reset models averaged score 0.6983 vs reference pair 0.7000 (-0.0017; below both
  reference singles). Refuted with margin.
- quant i0: -0.0002 single seed, inconclusive; seed-4 run queued for a pair comparison.

### R3: i1 cross-impact block (`--quant-i1`, +31 inputs) — negative (fold 0)
-0.0011 (SD 0.0010), class 3 -0.0040. Stopped before fold 1.

### R4: class-3 expert (`--train-classes 3`, routed on class 3) — negative (fold 0)
-0.0016 (SD 0.0006, P(<=0) 0.997): the joint model beats a class-3 specialist trained on fewer
sequences. Stopped before fold 1.

### R5: pooled weighted-Pearson + wMSE loss from scratch (`--loss wmse_gcorr`) — inside seed noise
+0.0003 vs reference seed 3 (SD 0.0005), -0.0006 vs reference seed 4. No evidence of an effect; not
replicated further.

### R6: dual-clock GRU (`--arch dual`, row GRU + event-clock GRU) in the EMA regime — no gain
All fold sequences: fold 0 0.7001 vs 0.7005, fold 1 0.6752 vs 0.6765.

### Engineering side result: C++ engine (branch `cpp-engine`, `native_cpp/`)
Parity 2.5e-6 vs PyTorch fp32 and 1.4e-6 vs the Rust engine. With fp16 dense weights on v3 it runs
hidden 128 at 17 us/row vs 27 (Rust) on this host, hidden 256 at 85 vs 156; fp16 changes WP by -1e-6 on
60 sequences. A faster member means more members or a wider model inside the 60-minute budget.

### Protocol fix: fixed final epoch
`train_rnn.py` saves `model.pt` as the best epoch **on the held-out fold** (selection on the evaluation
data). Only the hidden-256 runs peaked before the last epoch; their predictions were redone from
`model_ep3.pt`, and `fold_job.sh` now always predicts from the fixed final epoch (`EVAL_EPOCH`, default 3).

### R7: capacity, hidden 256 (EMA-4 recipe) — small gain, not worth its cost
Fixed final epoch: +0.0006 vs the seed-3 reference (SD 0.0010; fold 0 +0.0007, fold 1 +0.0005;
classes 1/2 +0.0023/+0.0027, class 3 +0.0003). Epoch curve: +0.003-0.004 after epoch 0, shrinking to
+0.0005 by epoch 3, i.e. the EMA-4 hidden-128 model catches up. One hidden-256 member costs ~4x a
hidden-128 member in inference and scores -0.0006 below the two-seed hidden-128 pair (2x cost).
Seed-4 hidden-256 runs are queued to check a hidden-256 pair.

### R2 follow-up: order-flow features, seed pair vs reference seed pair — refuted
quant(s3)+quant(s4) vs ref(s3)+ref(s4): **-0.0009** (SD 0.0006, P(<=0) 0.96), fold 0 -0.0007, fold 1
-0.0011, every class negative (class 3 -0.0009). Seed 4 alone: -0.0016. The single-seed class-3 hint was
noise; the explicit window sums slightly hurt (extra inputs, no new information).
