# Idea and combination queue (2026-10-01, updated by the 20-minute loop)

Ranked by expected public gain per GPU-minute. Status: queued / running / pass / reject.

## Member families (ensemble ingredients, own code)
| id | ingredient | evidence so far | status |
|---|---|---|---|
| F1 | since GRU128 pair + GCLP (incumbent) | public 0.6622; clean local 0.68952 | anchor |
| F2 | distilled GRU256 since student (α .5, fold teachers) | teammates: +0.0013–0.0016 over plain h256, class 3 +0.0038 | fold 0 running |
| F3 | LSTM128 since member (seed 5) — Jane Street 6th seed-bag idea, mixed cells | none | deprioritised (no native LSTM; head bag gives cheaper diversity) |
| F4 | chunk-500 GRU128 (final ab84 exists; base recipe) | old folds +0.0013, holdout −0.00003 | needs clean-fold check |
| F5 | extra GRU128 since seeds (seed bag) | teammates' curve: 3rd +0.00025, 4th +0.00012 | cheap, low value |
| F6 | long-BPTT since member (chunk 5000, batch 12) targeting per-sequence regime/scale headroom (oracle +0.017) | none; recipe confound (batch) | idea |

## Training/FT mechanisms
| id | mechanism | status |
|---|---|---|
| N1 | joint ensemble head FT (vs matched solo) | REJECT E-293 (−0.0001 vs solo) |
| N2 | pooled-memory correlation FT loss (decay .995) | REJECT E-291 (−0.0015) |
| N3 | cross-fitted convex blend weights | REJECT E-292 (−0.0004) |
| N4 | head-only vs two-stage FT | head-only wins (d256 +0.0025 single; since pair +0.0006) |
| N5 | shared-trunk head bag: parent head + head-only FT head (exact merge to one wide head) | PASS folds 0+1 +0.00098, holdout +0.00038; f2/f3 confirm queued |
| N6 | distilled head (head FT toward 0.5y+0.5 teachers) as 3rd head | queued |
| N7 | wMSE head-FT as extra head (final members, holdout) | export queued |

## Combinations to try once components pass
- C1: F1 + F2 (+ N3 weights) — primary candidate.
- C2: F1 + F2 + F3 — mixed GRU/LSTM bag (needs native LSTM from engine agent).
- C3: N1 applied to the final member set (only if N1 beats solo on folds 0+1).
- C4: N2 applied to the student's FT (only if N2 beats batch gcorr on the pair).
- C5: F2 teacher = pair + LSTM (larger teacher bag) for class-3 denoising — only if F3 passes.

## Rejected tonight
- E-287 current-code spread labels (+0.00002 vs matched control).
- E-288 per-class output scale (already calibrated; bound +0.00006).
- Bid/ask mirror augmentation: not identifiable (per-column anonymised codes; asymmetric target means
  t0 −0.57 / t1 −0.20 in a sampled train sequence). Not run.
- Online refit from the Kaggle notebook: impossible here (no labels at test time).
