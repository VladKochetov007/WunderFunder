# Run card: ab46_ref_f0

## Config

- name: ab46_ref_f0
- train_source: train+valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
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
| 0 | 0.6665 | 0.6632 | 0.6697 | 0.799 | 0.7477 | 0.7413 | 0.4357 | vfold468 |
| 1 | 0.6724 | 0.6657 | 0.6791 | 0.7993 | 0.7518 | 0.7482 | 0.4424 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
