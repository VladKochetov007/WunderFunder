# Run card: ab22_heur_e2

## Config

- name: ab22_heur_e2
- train_source: train
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6633 | 0.6575 | 0.6691 | 0.7742 | 0.7499 | 0.7157 | 0.4405 | devB764 |
| 1 | 0.6705 | 0.6632 | 0.6778 | 0.7812 | 0.7559 | 0.7203 | 0.4463 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
