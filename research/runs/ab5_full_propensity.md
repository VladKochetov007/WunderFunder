# Run card: ab5_full_propensity

## Config

- name: ab5_full_propensity
- train_source: train
- eval_source: devB
- seed: 3
- epochs: 1
- lr: 0.001
- wd: 0.01
- propensity: prop_v1
- loss: mix
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6290 | nan | nan | 0.7807 | 0.7536 | 0.7184 | 0.433 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
