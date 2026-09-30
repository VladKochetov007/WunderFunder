# Run card: ab26_tv_f0

## Config

- name: ab26_tv_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6607 | 0.6520 | 0.6694 | 0.7905 | 0.7379 | 0.7324 | 0.4284 | vfold468 |
| 1 | 0.6692 | 0.6617 | 0.6767 | 0.7987 | 0.7474 | 0.7401 | 0.4364 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
