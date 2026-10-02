# Teammate results intake (ideas and numbers only)

Recorded 2026-10-01 02:10 Kyiv from `origin/quant-probes`, `origin/cpp-engine` (author "nb") and
`origin/research/creative-iteration-20260930`, `origin/archive/2026-10-01` (teammate A), fetched
read-only. **Clean-room rule:** nobody implementing our code reads teammate source files; we reuse ideas and
reported numbers only. Our code must differ in structure, names and format. Their branches are never merged.
All numbers below are teammate-reported and not re-verified by us unless an E-id says so.

## Public leaderboard (teammate packages)

| package | members | public | runtime |
|---|---|---:|---|
| V1 | candC parents, 2-epoch head FT | 0.6641 | 41m42s |
| candC | distilled GRU256 + GRU128 s3 + GRU128 s4 + GRU128 chunk-500, equal 1/4 | 0.6640 | 48m52s |
| candB | GRU256 + 2×GRU128, head FT, 1/3 each | 0.6639 | 43m |
| candA | 2×GRU256 (runtime governor dropped one member) | 0.6629 | 35m |
| V3 | candC members, parents on train only | 0.6629 | 49m26s |
| V2 | candC members, valid weighted ×3 in parents | 0.6611 | 41m42s |

Ours: incumbent 0.6622 (2×GRU128 since + GCLP). Leader reading 0.6677 (29 Sep evening). Public comparison
noise about ±0.0006.

## Recipe facts
- Parents: train + 1× all valid, wMSE, aux struct/next 0.3, propensity² (eps .03), EMA teacher, since-weight
  τ32 floor .25, fixed epoch 3. FT: head only (input projection and GRU frozen), gcorr, batch 16, 1 epoch,
  lr 5e-4, all valid.
- Distilled student target (1−α)y + α·mean(teachers), α=0.5, teachers = two GRU256 parents.
- Inference: their own fp16 engine; about 2.4× faster scorer than their local simulation.

## Local evidence reported (their chain folds, "pure" sequences)
- GRU256 + head-only FT vs champion recipe: +0.0007 on 4 folds (+0.0009 folds 2+3); full FT erases the gain.
- Distilled GRU256 over plain GRU256: +0.0016 (s3), +0.0013 (s4); class 3 +0.0038. Distilled GRU128: +0.0010
  single but +0.0002 as a pair.
- Chunk-500 GRU128 as 4th member +0.00035. Cross-fitted global blend weights +0.00035; per-class weights 0.
- Seed saturation (members 1→6): +0.00074, +0.00025, +0.00012, +0.00007, +0.00005.
- Averaging all 22 fine-tuned members: only +0.0004 over candC (ensemble ceiling of these families).
- Training on valid-like data worth about +0.003 locally; parents with valid 0×/1×/3× → public .6629/.6640/.6611.

## Refuted (do not repeat)
Causal XL-Transformer (in 0.8/0.2 blend +0.0001, fails gate), event-token attention GRU, two-stream GRU,
V-REx, class-3 expert/atom/context heads, order-flow "quant" features, volume-level aug/drop, periodic state
reset, pooled-Pearson+wMSE from scratch, dual-clock GRU, EMA-6, h384, gen-2 distillation, 6-epoch distill,
privileged non-causal teachers (no better than ordinary distillation), per-class blend weights, online-norm
4th member, matured labels (was leakage), decoded class-3 tick features (|corr| ≤ 0.044), sparse volume
flow coordinates.

## Their untested ideas (ranked by them)
1. Auxiliary losses off during head FT. 2. Ensemble-aware head FT (each member's effective scale inside the
raw average is never trained). 3. Event GRU fed by the row GRU state. 4. Trade-mark surprise feature.

## Data findings (to verify independently)
- Targets change only on i0 events that move ask or bid (≥99.98%).
- Spread term is an exact function of the 22 price codes when both targets ≤ 0.
- Loser ≈ −(sp + 2m), m = loser's first adverse move clipped to [1, winner] (99.8/99.3/94.5/86.0% classes 3/2/1/0).
- Winner ≈ round(mean mid over the next N quote-move events), N = 4/6/6/8.
- Oracle per-sequence level for D=(t0−t1)/2 → 0.727; per-sequence scale-only oracle → 0.719 (upper bounds).
- Single-sequence chains: 64% class 3, score 0.59 vs 0.74 for long chains; suspected class-3-heavy, shorter
  public test.
