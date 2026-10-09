# Fold-1 seed3/seed4 baseline-pair grade R1

Frozen spec SHA-256: `f6de6f79fad019300ce95a5bcd96fadc1fdeaceae19ae272be76f6d88258428b`.
Metric: `global_wp_clip2_v1`; fold 1 has 457 exposed groups and 149 connected components (812,167 selected rows).

| Endpoint | WP | t0 | t1 |
|---|---:|---:|---:|
| Seed 3 | 0.657638438 | 0.650484890 | 0.664791985 |
| Seed 4 | 0.656866806 | 0.650759785 | 0.662973828 |
| Raw equal mean (1:1) | 0.659150436 | 0.652628188 | 0.665672685 |

Primary contrast: equal mean minus seed 3.
Paired whole-component bootstrap: +0.001511999 WP; 95% percentile CI [-0.000366105, +0.003279154] across 149 components (2000 replicates, seed 271828).

This is a baseline diagnostic on a historically inspected fold, not a fresh holdout, new-best claim, or public-score estimate. Component resampling does not include training-seed variation or public transfer.

Machine-readable result: `scratch/fold1_gru384_successor_20261009/grading_r1/BASELINE_PAIR_R1.json` (920981596b67d7115a8fc476eed7d10eb9784debb7f5ed38eb7042f3379fc224).
Centered moments: `scratch/fold1_gru384_successor_20261009/grading_r1/BASELINE_PAIR_R1.moments.npz` (1d5a2784781ddb9dfa91039cef681c4cad519b339a7e054ad3a05e374a29178e).
