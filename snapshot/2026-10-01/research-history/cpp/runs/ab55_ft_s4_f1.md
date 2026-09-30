# Run card: ab55_ft_s4_f1

## Config

- name: ab55_ft_s4_f1
- train_source: valid
- valid_fold: 1
- eval_source: vfold
- seed: 4
- epochs: 1
- lr: 0.0002
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- init: ab29_mask_f1
- freeze: 
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6561 | 0.6486 | 0.6635 | 0.7905 | 0.7544 | 0.726 | 0.4344 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab55_ft_s4 - ab39_ftm lam=0.0: 0: -0.0005±0.0008 (0.6708 vs 0.6713)  1: -0.0012±0.0006 (0.6561 vs 0.6572)  pooled: -0.0008±0.0004 (0.6637 vs 0.6645)`
- `ab55_ft_s4 - ab39_ftm lam=0.02: 0: -0.0004±0.0007 (0.6494 vs 0.6499)  1: -0.0011±0.0005 (0.6334 vs 0.6344)  pooled: -0.0007±0.0004 (0.6416 vs 0.6424)`
