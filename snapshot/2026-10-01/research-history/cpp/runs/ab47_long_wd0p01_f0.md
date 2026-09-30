# Run card: ab47_long_wd0p01_f0

## Config

- name: ab47_long_wd0p01_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 12
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- loss: wmse
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6648 | 0.6602 | 0.6694 | 0.7984 | 0.7437 | 0.7351 | 0.4311 | vfold468 |
| 1 | 0.6649 | 0.6537 | 0.6761 | 0.7902 | 0.7468 | 0.7429 | 0.4375 | vfold468 |
| 2 | 0.6696 | 0.6613 | 0.6779 | 0.7976 | 0.7471 | 0.7435 | 0.444 | vfold468 |
| 3 | 0.6697 | 0.6618 | 0.6775 | 0.7907 | 0.7543 | 0.748 | 0.4446 | vfold468 |
| 4 | 0.6734 | 0.6674 | 0.6794 | 0.8014 | 0.7518 | 0.7477 | 0.445 | vfold468 |
| 5 | 0.6710 | 0.6635 | 0.6784 | 0.801 | 0.7557 | 0.7483 | 0.4371 | vfold468 |
| 6 | 0.6723 | 0.6663 | 0.6782 | 0.8043 | 0.7544 | 0.7505 | 0.4384 | vfold468 |
| 7 | 0.6668 | 0.6580 | 0.6755 | 0.8032 | 0.7557 | 0.7481 | 0.4268 | vfold468 |
| 8 | 0.6594 | 0.6502 | 0.6686 | 0.8025 | 0.7551 | 0.7496 | 0.4242 | vfold468 |
| 9 | 0.6528 | 0.6466 | 0.6589 | 0.8043 | 0.7549 | 0.7506 | 0.4136 | vfold468 |
| 10 | 0.6476 | 0.6376 | 0.6577 | 0.8035 | 0.7549 | 0.749 | 0.4069 | vfold468 |
| 11 | 0.6462 | 0.6352 | 0.6572 | 0.8031 | 0.7549 | 0.7486 | 0.4055 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
