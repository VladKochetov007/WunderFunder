# Run card: ab26_tv_f2

## Config

- name: ab26_tv_f2
- train_source: train+valid
- valid_fold: 2
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6858 | 0.6844 | 0.6871 | 0.7893 | 0.7744 | 0.6915 | 0.4529 | vfold443 |
| 1 | 0.6937 | 0.6941 | 0.6933 | 0.7948 | 0.7785 | 0.7001 | 0.4655 | vfold443 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
