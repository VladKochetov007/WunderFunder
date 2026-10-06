# Quiet stretches are not a hold problem

Frozen equal parent/head blend, full supplied validation, `global_wp_clip2_v1`. No fit. Script `wave1_stretch.py`, numbers in `wave1_stretch.json`. The blend scores **0.700771**, the same supplied-file number as wave 1. That is not a public forecast. Confirmed public best stays 0.6660.

Age here is rows since the last joint `i0` touch (the production AND clock). Horizon 8 is the stopped-bootstrap rule in `blended_target`: the original label leaves the regression target only when the next label change is more than 8 rows ahead and `t+8` is still inside the sequence.

| Slice | t0 weight share | WP | rows |
|---|---:|---:|---:|
| Age ≥ 32 | 0.389 | 0.5837 | 1,236,034 |
| Age ≥ 32 and next change > 8 | 0.300 | 0.4818 | 952,864 |
| Age ≥ 32 and next change ≤ 8 | 0.089 | 0.8588 | 283,170 |
| Opening touch of a stretch that reaches age 32 |  | 0.4259 | 15,977 |

Copying the opening-touch prediction onto the age ≥ 32 rows scores **0.2983** there (−0.285 versus the blend) and **0.6161** on the full file (−0.085). Holding every row at the last touch scores **0.5758**. The non-holdout split matches (far rows 0.482, near rows 0.862, held quiet rows 0.302).

The model is better on the late rows of a long stretch than on the touch that opened it. A carried event state is the wrong direction.

The recipe weight `clamp(1 / (1 + since_i0px / 32), min=0.25)` already puts 0.639 of t0's metric mass into the loss. Rows at the 0.25 floor are 0.198 of t0 weight. The far rows are both downweighted and, under horizon 8, trained against the teacher instead of `y`. That overlap is real. It does not show that putting `y` back would raise WP. Nothing in this file cleared +0.005.

## What the Oct 5 reviews get wrong

- Event-hold and "belief carried from the touch" lose on this champion. The near rows are already 0.86.
- Old age is the wrong cut. Age ≥ 32 rows that sit within 8 of the next label change are not the hole.
- E488 readout transfer, the B16 cadence capsule, and the reanchor capsule under `cap3g_*` were materialized and not run. A prepared draft is not a gain.
- Oracle headroom on a fitted panel is not a recoverable training target. The event-row prediction of a long stretch is already a poor match to that stretch's label.
