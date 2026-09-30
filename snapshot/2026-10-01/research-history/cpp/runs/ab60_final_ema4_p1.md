# Run card: ab60_final_ema4_p1

## Config

- name: ab60_final_ema4_p1
- train_source: train+valid
- valid_fold: -1
- eval_source: holdout
- seed: 4
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
| 0 | 0.6838 | 0.6707 | 0.6969 | 0.7871 | 0.7467 | 0.7156 | 0.4296 | holdout368 |
| 1 | 0.6920 | 0.6796 | 0.7044 | 0.7959 | 0.7547 | 0.7256 | 0.4402 | holdout368 |
| 2 | 0.6940 | 0.6821 | 0.7059 | 0.7981 | 0.7567 | 0.7299 | 0.4445 | holdout368 |
| 3 | 0.6946 | 0.6830 | 0.7062 | 0.7988 | 0.7572 | 0.7312 | 0.4454 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
