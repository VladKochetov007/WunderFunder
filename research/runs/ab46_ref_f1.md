# Run card: ab46_ref_f1

## Config

- name: ab46_ref_f1
- train_source: train+valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
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
| 0 | 0.6524 | 0.6448 | 0.6601 | 0.7862 | 0.7469 | 0.7177 | 0.426 | vfold457 |
| 1 | 0.6590 | 0.6521 | 0.6660 | 0.7911 | 0.7524 | 0.7266 | 0.4334 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
