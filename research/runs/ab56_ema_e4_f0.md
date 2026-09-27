# Run card: ab56_ema_e4_f0

## Config

- name: ab56_ema_e4_f0
- train_source: train+valid
- valid_fold: 0
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
| 0 | 0.6641 | 0.6585 | 0.6697 | 0.7923 | 0.7419 | 0.7369 | 0.4312 | vfold468 |
| 1 | 0.6733 | 0.6671 | 0.6794 | 0.8014 | 0.7511 | 0.7469 | 0.4425 | vfold468 |
| 2 | 0.6757 | 0.6695 | 0.6820 | 0.8036 | 0.7538 | 0.7499 | 0.4464 | vfold468 |
| 3 | 0.6761 | 0.6699 | 0.6824 | 0.8042 | 0.7546 | 0.7504 | 0.4466 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
