# Run card: ab21_final_tv_e2

## Config

- name: ab21_final_tv_e2
- train_source: train+valid
- eval_source: holdout
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
| 0 | 0.6841 | 0.6712 | 0.6969 | 0.7897 | 0.7481 | 0.7154 | 0.4222 | holdout368 |
| 1 | 0.6877 | 0.6745 | 0.7008 | 0.7925 | 0.749 | 0.7218 | 0.4378 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
