# Wave 1 — what the champion misses, measured

Frozen predictor: equal blend of `c20261004_combo_s4_final` epoch 3 and `c20261004_combo_head_s4_final` epoch 0, FP32 caches. Metric `global_wp_clip2_v1`. No holdout sequence was used to fit anything (`wnn.splits.is_holdout`). Scripts: `wave1_anatomy.py`, `wave1_quiet.py`, `wave1_book.py`. Raw numbers are the matching JSON files.

## The score

| Predictor | Full supplied validation WP |
|---|---:|
| Equal parent/head blend | **0.700771** (t0 0.6968, t1 0.7047) |
| Same blend, each output snapped to the class quantum | 0.695162 |
| Same blend, sum snapped onto the quantum lattice | 0.699442 |
| Predict zero | 0.000 |

The blend matches the October 5 scorecard to the printed digits. Snapping onto the tick lattice destroys information. The champion's off-lattice values are doing work.

Per class, same blend: class 0 **0.816**, class 1 **0.786**, class 2 **0.750**, class 3 **0.456**.

## The label clock, on all 3,528,273 scored validation rows

A label change means the target pair moved by more than 1e-5 from the previous row.

| Clock (`i0_p0`, `i0_p11`) | Label unchanged | Label changed | Change rate |
|---|---:|---:|---:|
| neither | 2,701,584 | 76 | 0.000028 |
| p0 only | 46,223 | 8 | 0.00017 |
| p11 only | 54,241 | 8 | 0.00015 |
| both | 307,840 | 418,293 | 0.576 |

418,293 of 418,385 label changes sit on the joint touch move. The production AND flag is the label clock. The OR-versus-AND argument does not identify a different set of target jumps. `a6` is not that clock either: it changes on 3,294,172 scored rows, including 2,631,255 rows with no i0 touch move.

Mean gap between OR events on the first 40 train sequences is **4.04 rows**. Typical rows are near an event. Metric weight is not.

## Where the champion's weight and score sit

Deployable cells are the OR clock crossed with age since that event. Shares are of `|clip(y, 2)|`. Local WP is inside the cell.

| Cell | t0 weight share | Local WP |
|---|---:|---:|
| age 1–7, no touch change | 0.251 | 0.789 |
| age 32–127, no touch change | 0.198 | 0.579 |
| age 8–31, no touch change | 0.194 | 0.682 |
| age 0, both touches changed | 0.179 | 0.833 |
| age 128+, no touch change | 0.144 | 0.450 |
| age 0, one touch only | 0.033 combined | ~0.63 |

The event row is the champion's best large cell. The miss is the old quiet book. Age ≥ 32 and no touch change is about **34% of t0 weight**. On non-holdout sequences that cell is 31.3% of scored rows, the incumbent scores **0.512** there, and **82.4% of its absolute-target weight is class 3**.

## What was added, and what it did

All corrections use alpha 1, the split `sha256(seq_ix:wave1b) % 2` on non-holdout validation, and are scored on the eval half only.

Global linear model (log age, AND/OR bits, five Hawkes compensators with β = 1/4.04, touch depletion, i1 displacement, touch imbalance, a6):

| Model on the eval half | WP |
|---|---:|
| Frozen blend | 0.69018 |
| Affine map of the blend | 0.68474 |
| Those features plus the blend | 0.68371 |
| Same features trained on 400 train sequences, no blend | 0.28279 |

The features do not pay. An affine recalibration of the blend also does not pay.

Inside the quiet age≥32 cell only (426,180 fit rows, 475,809 eval rows), features standardized on the fit cell:

| Model | WP on the cell | Delta vs blend |
|---|---:|---:|
| Frozen blend | 0.51204 | |
| Standardized features alone | 0.14121 | |
| Features plus blend, ridge | 0.50060 | **−0.01144** |
| Width-32 residual MLP, 4 passes, 200k fit rows, zero-init output | 0.51220 | **+0.00016** |

Largest weighted correlations with the blend's residual on that cell: touch imbalance +0.034/+0.064, a6 level +0.060/+0.054, trade compensator −0.056/−0.050. i1 displacement is −0.008/+0.005. Event bits are identically zero inside the cell, which is the sanity check.

Replacing those hand-built features with all 44 i0 volume columns and all 44 i1 volume columns, same cell and split: ridge delta **−0.01543**. Strongest single column residual correlation is `i0_v17` at about −0.066.

## What this kills

- Tick-lattice projection of the champion.
- `a6` as the event bit.
- Single-sided touch moves as a missed label clock.
- A linear Hawkes / depletion / i1 / age correction, globally or only on the expensive cell.
- A small residual network on that same state.
- A linear read of the full volume book on that cell.
- The hope that i1's displacement since the last i0 move is unused alpha against this champion.

## What is still open, and what it is worth

The quiet class-3 stretch is a real hole in the score and a poor hole for another feature stack. The champion is already at 0.83 WP on the touch event and the residual correlations left in the hole are around 0.06. A complete-predictor gain the size of the October CV result (+0.0024) is about one fourteenth of the 0.034 gap from public 0.6660 to 0.7000. These probes do not move that gap.

Still untested, and not implied by a negative feature ridge:

- One training-protocol change: collapse each quiet stretch to a single weighted outcome, or replace per-chunk loss normalization with a pooled denominator. Historical size of the normalizer idea was a few ten-thousandths, not a few hundredths. Kill it if a matched tiny model does not gain at least +0.005 WP inside this quiet cell.
- Whether a second quantized member fits in the minutes left under the 60-minute cap. That is a runtime measurement, not an accuracy measurement.
- A public submission that changes low-propensity rows, if the question is whether the live mask is softer than `is_scored`. That spends a leaderboard slot and is a human decision.

Kaggle quota was read and not spent. TPU time used is about 135 seconds of the 20-hour weekly allowance, refresh 2026-10-10. GPU time used is about 10.3 hours. The API's GPU allowance field did not match the older 30-hour record, so no kernel was pushed on the back of a negative local probe.

## Not a forecast

Nothing here is an expected public score. The 0.7008 validation number is the same supplied file the scorecard already published, and 1,505 of its sequences entered the final fit.
