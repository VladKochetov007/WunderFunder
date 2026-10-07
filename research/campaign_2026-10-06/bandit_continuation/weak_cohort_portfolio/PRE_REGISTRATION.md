# E-513 / H-201: weak-model cohort as a residual portfolio

Registered 2026-10-07 01:05 UTC, before scoring any new cohort result. This is a CPU-only prediction analysis; it does not fit model weights.

## Question and tree placement

Can an equal-weight average of several individually weak but differently trained models cancel member-specific residual noise and become a better complement to a strong anchor than one weak member? This is a child of H-184's sparse-blend direction and Card 8's weak-model residual direction. E-177 already scanned weak models individually; this test isolates their joint averaging interaction. E-177 also cautions that strength usually matters more than raw decorrelation, so the child has no positive numerical prior and stays behind active C1 grading and the cloud teacher path.

## Frozen members and data

Use only the following fold-specific validation predictions, each on rows held out from that member's fit:

| Role | Fold 0 member | Fold 1 member | Lineage |
|---|---|---|---|
| Strong anchor A | `ab56_ema_e4_f0` (`ab62_base_s3_f0` cache alias) | `ab58_ema_e4_f1` (`ab62_base_s3_f1` cache alias) | Independently scratch-trained EMA-4 GRUs; see run cards and E-170/E-177 |
| Weak cohort member 1 | `ab48_basehead_f0` | `ab48_basehead_f1` | Fine-tuned from the organizer-provided baseline, with input/recurrent core frozen; public supplied baseline, not a teammate checkpoint |
| Weak cohort member 2 | `ab62_td5_f0` | `ab62_td5_f1` | Independently scratch-trained TD-target model; config/log records exist, but no modern run card or prediction sidecar |
| Weak cohort member 3 | `ab53_p2_f0` | `ab53_p2_f1` | Independently scratch-trained two-epoch prop-v2 GRUs; run cards exist |

The fixed weak portfolio `C` is the arithmetic mean of those three members' raw predictions. The anchor/portfolio endpoint is `(1 - λ) A + λ C`, with `λ ∈ {0, .05, .10, .15, .20, .25, .30, .35, .40}`. Mix raw member outputs and apply the official scorer's `[-2, 2]` clipping to the final prediction. Do not reweight members, search a larger pool, add a residual readout, or change the score mask in E-513.

## Calibration and comparison

Use only fold-specific out-of-fold predictions on the corresponding `fold == k` groups. Exclude every validation group whose connected chain contains a training group, following E-194's pure-chain definition. For the first direction, choose `λ` by maximum global `global_wp_clip2_v1` on pure fold 0 and evaluate that exact `λ` on pure fold 1. Repeat with fold 1 as calibration and fold 0 as evaluation. Ties within `1e-8` choose the smaller `λ`.

The primary comparison is the held-out delta of the fixed three-member cohort mix versus A. A single-member control uses `ab48_basehead` only, with its own `λ` selected on the same calibration fold and evaluated on the other fold. Report both directions and target channels separately; do not average fold scores or describe the two fitted values as one deployable global blend. Show the equal-weight cohort alone as a descriptive diagnostic.

For uncertainty, resample pure validation chains with replacement, paired between the selected cohort mix and A, and recompute pooled clipped weighted Pearson on each resample. Use 2,000 deterministic bootstrap draws and report intervals conditional on the calibration-selected `λ`; do not refit `λ` inside these test-fold intervals.

## Provenance and stop gates

Before cohort scoring, hash the target cache, chain-fold map, all six prediction arrays, all source checkpoints, run configs/cards/logs, and the metric/prediction code. Confirm array shapes, finite selected predictions, exact target-cache row-group order, fold membership counts, and fold-specific run membership. The legacy prediction files have no per-array JSON sidecars. Their row alignment must be reconstructed from the common `valid_targets` cache contract and checked by reproducing the recorded anchor and `ab53_p2` run-card fold scores; record the remaining provenance limit. If either check fails materially or hashes/config lineage cannot be recovered, stop before calculating the cohort comparison and report the blocker.

All folds have been inspected before; the E-177 scan and associated selection evidence also used these development folds. Therefore E-513 is retrospective exploration, not independent confirmation, a fresh holdout, a leaderboard forecast, or grounds to add a local delta to public score. Promote only the mechanism question to a new, separately preregistered Kaggle cohort if the joint portfolio beats both A and the fixed single-member control in both directions with paired-chain uncertainty compatible with the improvement. A negative result lowers priority for this exact cohort recipe; it does not erase other weak-model uses such as teacher targets or representations.

## Expected artifacts

`weak_cohort_eda.py` (live marimo notebook), `plots/`, immutable source/prediction manifest, result JSON, and a short report. Local experiment compute stops at 10:00 Europe/Kyiv today; any later GPU/TPU work uses Kaggle only.
