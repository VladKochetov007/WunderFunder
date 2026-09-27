# Run card: ab61_linear_lstm_f0

## Config

- name: ab61_linear_lstm_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- enc: linear
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
| 0 | 0.6655 | 0.6605 | 0.6706 | 0.798 | 0.7495 | 0.7398 | 0.422 | vfold468 |
| 1 | 0.6701 | 0.6630 | 0.6773 | 0.7999 | 0.7505 | 0.7451 | 0.4348 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
