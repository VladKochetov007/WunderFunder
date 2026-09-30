# Run card: ab61_mlp_lstm_f0

## Config

- name: ab61_mlp_lstm_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- enc: mlp
- cell: lstm
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
| 0 | 0.6608 | 0.6545 | 0.6671 | 0.7971 | 0.7426 | 0.7405 | 0.4219 | vfold468 |
| 1 | 0.6711 | 0.6654 | 0.6767 | 0.8023 | 0.7503 | 0.7428 | 0.436 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
