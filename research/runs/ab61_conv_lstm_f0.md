# Run card: ab61_conv_lstm_f0

## Config

- name: ab61_conv_lstm_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- enc: conv
- cell: lstm
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- fast: False
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6574 | 0.6466 | 0.6682 | 0.7953 | 0.7479 | 0.736 | 0.4195 | vfold468 |
| 1 | 0.6683 | 0.6614 | 0.6753 | 0.8013 | 0.7507 | 0.7442 | 0.4328 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
