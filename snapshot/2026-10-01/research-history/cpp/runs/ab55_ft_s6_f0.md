# Run card: ab55_ft_s6_f0

## Config

- name: ab55_ft_s6_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 6
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
- init: ab29_mask_f0
- freeze: 
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6707 | 0.6638 | 0.6777 | 0.802 | 0.753 | 0.7467 | 0.4368 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
