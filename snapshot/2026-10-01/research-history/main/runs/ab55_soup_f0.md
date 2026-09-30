# Run card: ab55_soup_f0

## Config


## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab55_soup - ab39_ftm lam=0.0: 0: -0.0000±0.0004 (0.6713 vs 0.6713)  1: -0.0002±0.0004 (0.6570 vs 0.6572)  pooled: -0.0001±0.0003 (0.6644 vs 0.6645)`
- `ab55_soup - ab39_ftm lam=0.02: 0: +0.0000±0.0004 (0.6499 vs 0.6499)  1: -0.0002±0.0003 (0.6343 vs 0.6344)  pooled: -0.0000±0.0003 (0.6423 vs 0.6424)`
- `ab55_soup - ab29_mask lam=0.0: 0: +0.0017±0.0009 (0.6713 vs 0.6696)  1: +0.0002±0.0011 (0.6570 vs 0.6568)  pooled: +0.0010±0.0008 (0.6644 vs 0.6634)`
- `ab55_soup - ab29_mask lam=0.02: 0: +0.0013±0.0008 (0.6499 vs 0.6486)  1: -0.0003±0.0011 (0.6343 vs 0.6346)  pooled: +0.0005±0.0008 (0.6423 vs 0.6418)`
