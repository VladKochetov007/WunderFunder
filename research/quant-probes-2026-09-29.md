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

## Upload candidates (built 2026-09-29, not uploaded)
Members: `scripts/final_job.sh` (train + all 1,873 valid, EMA-4 + since-weight, then head-only gcorr FT on all
valid). Packaged with the C++ engine (`cpp-engine` branch, `native_cpp/package_cpp.py`, fp16 on v3), each
member checked against PyTorch (max abs 2.7e-4 with fp16 weights), run in the scorer-image userland.

| zip | members | sha256 | local sim us/row | projected scorer time* |
|---|---|---|---:|---:|
| candB_h256_2h128 | fin256_s3h, fin128_s3h, fin128_s4h (1/3 each) | 67dbdb63...23e9 | 168 | ~47 min |
| candA_h256pair | fin256_s3h, fin256_s4h (1/2 each) | 5e7f7d91...ea3a | 201 | ~56 min (governor drops one member) |

*Scorer ~2.36x faster than the local simulation (Rust pair: 25m52s public vs 96.9 us/row here, re-measured
the same hour). Upload B first; its public runtime calibrates whether A fits.
Fold evidence for the recipe: h256 head-FT pair vs champion analog +0.0007 (all four folds non-negative);
h256 + 2 x h128 kept ~70% of the pair gain in the fold-0/1 mixture study (parents, no FT).

# Round 2 (2026-09-29 evening): after public 0.6639 (candB), leader 0.6677

Public: candB (h256 + 2 x h128, head-FT) **0.6639** in 43 min (new best, +0.0017 over 0.6622); candA
(2 x h256) 0.6629 in 35 min: the governor dropped one member early, so it ran mostly as a single h256.

## EDA (OOF, 4 folds, candB analog 0.70082)
- Per-class output scale, cross-fitted: -0.0001 (scales ~1.0): the ensemble is calibrated across classes.
- Oracle per-sequence linear recalibration 0.744, scale-only 0.719: the headroom is in the per-sequence
  level of the **future move** D = (t0-t1)/2 (oracle D level 0.727; class-3 level R^2 0.39, predicted
  level sd half the target's). The half-sum level is not a lever (oracle lowers WP under |y| weights).
- Long-memory causal features (expanding z, EMA deviations hl 500-8000, 1k/5k changes of i0/i1 mid codes and
  activity clocks) do not predict the residual: |weighted corr| < 0.04, cross-fitted ridge negative in
  every class. The per-sequence drift behaves like a random walk.
- Train-unlike terciles (adversarial score, cv AUC ~0.99 per class): width and fine-tune gains are not
  concentrated on train-unlike sequences; the larger public than fold gain is within LB noise.

## Arms
- EMA-6 hidden 256: overfits after epoch 3 (fold 0 0.7029 at ep3 -> 0.7011 ep5; fold 1 0.6774 -> 0.6732);
  FT on the ep5 parent 0.6964/0.6687. Refuted; seed-4 runs stopped.
- Hidden 384 (fold 0, seed 3): parent +0.0005..+0.0010 over h256, after head-FT -0.0003..-0.0006. Stop.
- Class-conditional residual heads (h128, head-FT, fold 0 pair): +0.0003 (SD 0.0004), all classes small +.
- **Distillation** (student trained on 0.5 y + 0.5 mean(h256 teachers)), fold 0 seed 3, head-FT:
  h128 student vs h128 +0.0010 (SD 0.0005); vs a single h256 +0.0007 with the class split
  0/1/2/3 = -0.0016/-0.0013/-0.0023/**+0.0052**: distillation fixes class 3 (label noise), width helps
  classes 0-2. Seed-4 students and hidden-256 students queued.

## Round 2 results (folds 0+1, pure chains, head-FT members)
| arm | comparison | delta | fold 0 / 1 | note |
|---|---|---:|---|---|
| class-conditional heads (h128 pair) | vs h128 pair | -0.0001 | +0.0003 / +0.0001 pooled differently | refuted |
| distilled h128 student pair | vs h128 pair | +0.0002 | +0.0006 / -0.0001 | weak |
| online-norm h128 as 4th member | B + on128 vs B (fold 0) | -0.0002 | | refuted |
| chunk-500 h128 as 4th member | B + c500 vs B | **+0.00035** | +0.0004 / +0.0003 | SD 0.0002 |
| **distilled h256 student** (teachers h256 s3+s4, alpha 0.5) | vs h256 s3 | **+0.0016** | +0.0015 / +0.0014 | every class +, class 3 +0.0038 |
| | vs h256 s4 | +0.0013 | +0.0013 / +0.0006 | |
| B with the d256 student instead of h256 | vs B | +0.0004 | +0.0005 / +0.0004 | SD 0.0002 |
| **C = d256 + 2 x h128 + c500** | vs B | **+0.00065** | +0.0007 / +0.0006 | SD 0.0002; ~+1 h128 of runtime |

Adding more plain h128 seeds to B changes nothing (+0.00002); a second h256 in place of the h128s +0.00004.
Final C members: find256_s3 (distilled from fin256_s3/s4 on train + all valid), fin128_s3/s4, finc500b_s3.

## Upload candidate C (built 2026-09-29 evening, not uploaded)
`candC_d256_2h128_c500.zip` (7.4 MB): find256_s3h (hidden 256 distilled from fin256_s3/s4, head-FT), fin128_s3h,
fin128_s4h, finc500b_s3h (chunk-500 h128), 1/4 each, members in that order (the governor drops from the end).
Parity vs PyTorch with fp16 weights: 2.4e-4 (d256), 4.2e-4 (c500). Scorer-image simulation: 166.8 us/row vs
146.8 for candB in the same session -> ~49 min projected from candB's measured 43 min (governor budget 52).
Fold expectation: +0.00065 over candB (folds 0/1 +0.0007/+0.0006, SD 0.0002).

# Round 3 (2026-09-30 night): bolder ideas
Three literature subagents (LOB microstructure; noisy labels / distillation / sequence models; competition
winners and quant practice) proposed 15 ideas; triage: mirror symmetry is impossible here (price codes are not
bid/ask symmetric: class-3 bid codes in [-2.27,-1.27], ask in [-0.84,-0.46]); online matured-label correction ~
Vlad's refuted E-080; distributional/CE heads ~ refuted E-071; full-network FT hurts h256 (above).

## Privileged (non-causal) teacher distillation — no gain beyond ordinary distillation
Cross-fitted h128 teachers on train halves that also see raw rows t+L (`--lookahead`, `scripts/teacher_job.sh`,
`scripts/merge_teachers.py`). Teacher OOF valid WP: t+10,t+40 **0.813**; t+3,t+10 0.744 (causal ~0.70).
h128 students, fold 0, seed 3, head-FT, vs the h128 reference: t+10/40 alpha 0.5 +0.0007, alpha 0.8 +0.0011,
t+3/10 alpha 0.5 +0.0002; ordinary distillation from two causal h256 teachers +0.0010. A far more accurate
teacher does not make a better causal student: the distillation gain is smoothing, capped by what the causal
student can represent.

## Blend weights (8 OOF members, cross-fitted between folds 0/1)
Global weights +0.00035 over equal weights on both folds (weights ~2/3 on the two d256 students); per-class
weights +0.0000. Per-class blending refuted.

## Class-3 context table (Bayes-limit diagnostic, `scripts/c3_table.py`)
E_w[D | book-state code, log2 rows since last price event, last move sign, trade row] fitted on 1,500 class-3
train sequences: weighted corr with D on class-3 pure valid 0.157 / 0.127 (folds 0/1) vs the d256 pair 0.507 /
0.476; adding the table to the GRU lowers corr at every beta. No discrete context structure is left unused.

## Born-again generation 2 and longer distillation — plateau
Gen-2 h256 students, teacher = mean of four h256 models (two gen-1 students + two plain), alpha 0.7:
pair vs gen-1 pair +0.00036 (folds +0.0002/+0.0005, SD 0.0003); gen-2 in place of the gen-1 student in the C
blend -0.00006. Six epochs with distillation targets (fold 0): parent 0.7027 vs 0.7028 at four epochs (no
overfitting now, but no gain either).

## Round-3 verdict
No breakthrough. Distillation is the only lever and it plateaus after one generation; privileged teachers,
per-class blending, discrete class-3 context tables and longer training add nothing measurable. The class-3
information ceiling for causal models of this family looks reached; remaining ensemble gains are ~+0.0003.

# Public: candC 0.6640 in 48m52s (candB 0.6639): the +0.0007 fold gain is invisible on the public LB.

## New architecture: GRU + causal attention over the last 32 i0 price-event tokens (`--arch attn`)
`src/wnn/event_attn.py`: tokens (row projection + log gap) written at each i0 price event into a ring carried
across chunks; each row queries it from the GRU output (4 heads, d 64) and the head sees [GRU | proj | ctx].
Checked: chunked == full forward exactly, future perturbation leaves the prefix bit-identical.
h128, 4 epochs EMA: +0.0017/+0.0022/+0.0007 after epoch 0, final parents 0.7002 vs 0.7005 (f0), 0.6765 vs 0.6765
(f1). Head-FT: single -0.0002 (folds 0+1); fold-0 pair +0.0003 (SD 0.0004); as the 4th member in C instead of
c500 -0.0004. Refuted: faster convergence to the same ceiling, like width, distillation and privileged teachers.

## Decoded class-3 price level in ticks (`scripts/eda_ticks.py`) — no residual information
Transition tables fitted on 1,500 class-3 train sequences (1,594 transitions; 18 unknown of 1.96M valid book
events). Features on the decoded mid path: expanding z, distance to rolling max/min (500/2000/8000 rows),
tick changes over 100/500/2000 rows, reversal fraction over the last 64 mid moves. Weighted corr with the
d256-pair residual on class-3 pure valid rows: |corr| <= 0.044 with signs flipping between folds; cross-fitted
ridge -0.012 / -0.006. No support/resistance, momentum or reversal-regime signal beyond the GRU.

# Status after round 3
Every direction tried since the 0.6640 submission converges to the same ceiling (~0.702 on folds 0+1 for the
best 4-member blend): width, distillation (incl. privileged and gen-2), attention over events, class heads,
per-class blending, discrete class-3 contexts, long-memory and decoded-level features.

## Shift robustness (train-only protocol, `scripts/trainonly_job.sh`)
Train on train only, score all pure validation chains (streams never seen): reference 0.69566 (vs ~0.699 for
fold models that saw other validation chains: valid-like data is worth ~+0.003).
- `--level-aug 0.5` (per-sequence offsets on absolute volume codes): -0.0005 (SD 0.0005), class 3 -0.0023.
- `--level-drop 0.5` (zero the absolute volume block): holdout 0.6897 vs 0.6906. Both refuted.

## LB-only candidates (folds cannot measure test shift), built on C's architecture
- `candV1_ft2.zip` (sha ea494921...319d): C's parents with a two-epoch head fine-tune.
- `candV2_vr3.zip` (sha b13dfe87...9c75): all four members retrained with validation sequences repeated 3x.
Both ~C's runtime (~49 min).

## LB 2026-09-30: V1 0.6641 (41m42s), **V2 0.6611 (-0.0029)**, both 41m42s (C 48m52s: scorer speed varies ~15%)
Upweighting validation 3x in parent training clearly hurts the test; a longer head fine-tune is neutral.
Next LB probe: `candV3_trainonly.zip` (sha f7943bc6...0fe): C's four members with parents trained on TRAIN ONLY
(`PARENT_SOURCE_MODE=train scripts/final_job.sh`), validation used only by the head fine-tune.
