# Run card: ab47_long_wd0p1_f0

## Config

- name: ab47_long_wd0p1_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 12
- lr: 0.001
- wd: 0.1
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
| 0 | 0.6639 | 0.6591 | 0.6687 | 0.7978 | 0.7422 | 0.7342 | 0.4305 | vfold468 |
| 1 | 0.6645 | 0.6530 | 0.6759 | 0.79 | 0.7466 | 0.7421 | 0.4344 | vfold468 |
| 2 | 0.6681 | 0.6590 | 0.6771 | 0.7976 | 0.747 | 0.7428 | 0.4417 | vfold468 |
| 3 | 0.6680 | 0.6594 | 0.6767 | 0.7907 | 0.7553 | 0.7464 | 0.4444 | vfold468 |
| 4 | 0.6743 | 0.6693 | 0.6793 | 0.8042 | 0.7529 | 0.7479 | 0.4426 | vfold468 |
| 5 | 0.6740 | 0.6673 | 0.6806 | 0.8043 | 0.7565 | 0.7495 | 0.4395 | vfold468 |
| 6 | 0.6757 | 0.6705 | 0.6809 | 0.8044 | 0.7562 | 0.7492 | 0.4427 | vfold468 |
| 7 | 0.6754 | 0.6700 | 0.6809 | 0.8064 | 0.7585 | 0.7484 | 0.4413 | vfold468 |
| 8 | 0.6749 | 0.6683 | 0.6814 | 0.807 | 0.7586 | 0.7513 | 0.4469 | vfold468 |
| 9 | 0.6746 | 0.6685 | 0.6807 | 0.8085 | 0.7587 | 0.7523 | 0.4472 | vfold468 |
| 10 | 0.6724 | 0.6650 | 0.6798 | 0.8078 | 0.7593 | 0.7519 | 0.4435 | vfold468 |
| 11 | 0.6723 | 0.6647 | 0.6799 | 0.8077 | 0.7594 | 0.7517 | 0.4433 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
