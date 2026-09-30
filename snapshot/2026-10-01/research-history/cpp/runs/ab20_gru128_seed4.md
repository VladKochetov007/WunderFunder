# Run card: ab20_gru128_seed4

## Config

- name: ab20_gru128_seed4
- train_source: train
- eval_source: devB
- seed: 4
- epochs: 1
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
| 0 | 0.6672 | 0.6599 | 0.6745 | 0.7758 | 0.7541 | 0.7168 | 0.4405 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
