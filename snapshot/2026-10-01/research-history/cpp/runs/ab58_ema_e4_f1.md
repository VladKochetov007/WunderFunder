# Run card: ab58_ema_e4_f1

## Config

- name: ab58_ema_e4_f1
- train_source: train+valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 4
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- ema_teacher: 1.0
- ema_decay: 0.999
- eval_ema: True
- loss: wmse
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6500 | 0.6425 | 0.6574 | 0.7844 | 0.7463 | 0.7166 | 0.4245 | vfold457 |
| 1 | 0.6592 | 0.6520 | 0.6663 | 0.791 | 0.7543 | 0.7258 | 0.434 | vfold457 |
| 2 | 0.6614 | 0.6545 | 0.6683 | 0.7928 | 0.7555 | 0.7287 | 0.4366 | vfold457 |
| 3 | 0.6623 | 0.6559 | 0.6688 | 0.7935 | 0.7556 | 0.7297 | 0.4371 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
