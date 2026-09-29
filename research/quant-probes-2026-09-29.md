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

### Seed-ensemble curve (reference recipe, hidden 128, seeds 3-6) — saturates by K=3
Mean over all K-subsets, pure chains, folds 0+1 (`scripts/seed_curve.py`):

| K | mean WP | gain vs K-1 | min-max over subsets |
|---:|---:|---:|---|
| 1 | 0.69892 | | 0.69828-0.69971 |
| 2 | 0.69967 | +0.00074 | 0.69930-0.70005 |
| 3 | 0.69992 | +0.00025 | 0.69965-0.70014 |
| 4 | 0.70004 | +0.00012 | |

The single-model spread over seeds (0.0014) is larger than every feature/architecture effect tested here.

### R7 follow-up: hidden-256 seed pair — the best arm so far
h256(s3)+h256(s4) vs ref(s3)+ref(s4): **+0.0008** (SD 0.0008, P(<=0) 0.17), fold 0 +0.0009, fold 1
+0.0006, every class positive (0 +0.0003, 1 +0.0022, 2 +0.0020, 3 +0.0007). Pair WP 0.70078 vs the
four-seed hidden-128 ensemble 0.70004: width beats more seeds at this point of the curve.
Cost: C++ fp16 engine, scorer-image userland, one core of this host: 188 us/row for the pair (123 min
for 39.4M rows here; ~49 min if the scorer is ~2.5x faster as the old public runtimes suggest), i.e. at
the edge of the 52-minute governor budget. The Rust engine (157 us per hidden-256 member) cannot fit it.
Hidden-192 pair queued as the cheaper middle point.

### Seed curve extended to six hidden-128 seeds (3-8)
| K | mean WP | gain vs K-1 |
|---:|---:|---:|
| 1 | 0.69872 | |
| 2 | 0.69946 | +0.00074 |
| 3 | 0.69971 | +0.00025 |
| 4 | 0.69984 | +0.00012 |
| 5 | 0.69991 | +0.00007 |
| 6 | 0.69996 | +0.00005 |

The hidden-128 seed ensemble plateaus at ~0.7000. The hidden-256 pair (0.70078) is **+0.0008** above all
six hidden-128 seeds averaged (SD 0.0006; fold 0 +0.0006, fold 1 +0.0011): after two members, width is
worth more than more seeds. A third hidden-256 seed is running to average over three different pairs.

### Width mixtures at a lower inference cost (equal weights, vs the hidden-128 seed pair)
| ensemble | WP | delta | fold 0 | fold 1 | C++ fp16 step cost (this host) |
|---|---:|---:|---:|---:|---:|
| h256 | 0.69945 | -0.00056 | -0.0001 | -0.0010 | 85 us |
| h256 + h128 | 0.70017 | +0.00017 | +0.0005 | -0.0002 | 102 us |
| h256 + 2 x h128 | 0.70058 | +0.00058 | +0.0007 | +0.0005 | 119 us |
| h256(s4) + 2 x h128 (seeds 5, 6) | 0.70038 | +0.00038 | +0.0006 | +0.0001 | 119 us |
| 2 x h256 | 0.70078 | +0.00078 | +0.0009 | +0.0006 | 170 us |
| 2 x h256 + h128 | 0.70083 | +0.00083 | +0.0010 | +0.0007 | 187 us |

Runtime calibration: the Rust two-member ZIP took 25m52s on the public scorer and 100 us/row in the local
scorer-image simulation, so the scorer is ~2.4x faster than that simulation. Projected scorer time for
39.4M rows: 2 x h256 ~50 min (at the 52-minute governor budget), h256 + 2 x h128 ~43 min, 2 x h128 ~16 min.

### R8: width replicated over three seeds — hidden 256 is the one supported lever
Averaged over every seed subset (`seed_curve.py`), pure chains, folds 0+1, fixed final epoch:

| members | hidden 128 (seeds 3-8) | hidden 192 (seeds 3-4) | hidden 256 (seeds 3-5) |
|---:|---:|---:|---:|
| 1 | 0.69872 | 0.69894 | **0.70005** |
| 2 | 0.69946 | 0.69970 | **0.70090** |
| 3 | 0.69971 | | **0.70118** |

Paired, chain bootstrap: h256 triple vs h128 triple **+0.0010** (SD 0.0008; fold 0 +0.0014, fold 1
+0.0008; classes +0.0003/+0.0022/+0.0025/+0.0016). The three h256 pairs vs the h128 pair: +0.0008,
+0.0007, +0.0012. Hidden 192 is indistinguishable from 128 (pair -0.0003, folds disagree), so the gain is
not a smooth function of width at this sample size; hidden 256 is.

## Summary for the next iteration
1. Every feature/architecture probe (state reset, order-flow sums, i1 flow, class-3 expert, dual clock,
   pooled-Pearson loss) is flat or negative once seed noise is handled with seed pairs.
2. Hidden 256 members are ~+0.0013 per model over hidden 128 in this recipe; two of them beat any number
   of hidden-128 seeds. Untested: hidden 256 with the fine-tune stage (LP-FT gcorr) and EMA-6, and on
   untouched folds 2/3.
3. Deployment: a hidden-256 pair needs the fp16 C++ engine (branch `cpp-engine`) and is projected at ~50
   minutes, at the governor budget; h256 + 2 x h128 (~43 min) keeps ~70% of the pair gain.

## Confirmation with the fine-tune stage and on untouched folds 2/3 (2026-09-29, day)
Fine-tune = the champion's two-stage gcorr LP-FT on validation chains outside the held-out fold
(`scripts/fold_ft.sh`); "head" = its first stage only (trunk frozen, lr 5e-4); "full" = both stages.
New propensity nets prop_f2/prop_f3 and parents ref/refs4/h256/h256s4 were trained for folds 2 and 3.

Folds 0+1 (seed pairs, pure chains):
| pair | no FT | head FT | full FT | full FT, stage-2 lr 5e-5 |
|---|---:|---:|---:|---:|
| hidden 128 | 0.70000 | 0.70107 | 0.70088 | |
| hidden 256 | 0.70078 | **0.70143** | 0.70070 | 0.70080 |

Full-network fine-tuning erases the hidden-256 gain even at a 4x lower lr; head-only keeps it. Seed triples
(3,4,5): h256 head-FT 0.70194 vs h128 head-FT 0.70124 (+0.0007, both folds).

Untouched folds 2+3 and all four folds (seed pairs 3+4):
| comparison | folds 2+3 | all folds | per fold 0/1/2/3 | SD (all) |
|---|---:|---:|---|---:|
| h256 parents vs h128 parents | +0.0009 | | f2 +0.0001, f3 +0.0019 | |
| h256 head-FT vs h128 head-FT | +0.0007 | +0.0005 | +0.0005/+0.0007/+0.0003/+0.0014 | 0.0007 |
| **h256 head-FT vs champion analog (h128 full FT)** | **+0.0009** | **+0.0007** | +0.0001/+0.0014/+0.0001/+0.0020 | 0.0007 |
| h128 head-FT vs h128 full FT | +0.0002 | +0.0002 | -0.0004/+0.0007/-0.0002/+0.0006 | 0.0002 |

Classes (all folds, h256 head-FT vs champion analog): 0 +0.0009, 1 +0.0017, 2 +0.0018, 3 +0.0003. On folds
2+3 class 3 is slightly negative (-0.0003); the gain is in classes 0-2.
Verdict: hidden-256 members with head-only fine-tuning are non-negative on every fold and ~+0.0007 over the
champion recipe on average; the size is comparable to one public-LB comparison's noise (~0.0006), so a
public upload is the remaining test. Final members (train + all valid, head FT on all valid):
`scripts/final_job.sh`.
