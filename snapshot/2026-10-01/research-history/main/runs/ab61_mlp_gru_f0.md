# Run card: ab61_mlp_gru_f0

## Config

- name: ab61_mlp_gru_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- enc: mlp
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6676 | 0.6587 | 0.6765 | 0.7971 | 0.7492 | 0.7457 | 0.4367 | vfold468 |
| 1 | 0.6735 | 0.6674 | 0.6796 | 0.8019 | 0.7524 | 0.7486 | 0.4413 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
