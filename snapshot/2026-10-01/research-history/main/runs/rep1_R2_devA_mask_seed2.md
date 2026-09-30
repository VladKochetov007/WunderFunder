# Run card: rep1_R2_devA_mask_seed2

## Config

- name: rep1_R2_devA_mask_seed2
- train_source: valid_devA
- eval_source: devB
- seed: 2
- epochs: 2
- lr: 0.0002
- wd: 0.01
- propensity: 
- loss: mix
- init: ab1_A_mix
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6271 | nan | nan | 0.7779 | 0.7531 | 0.717 | 0.4304 | devB764 |
| 1 | 0.6306 | nan | nan | 0.7795 | 0.7542 | 0.7195 | 0.4368 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
