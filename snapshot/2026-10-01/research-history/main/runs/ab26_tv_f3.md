# Run card: ab26_tv_f3

## Config

- name: ab26_tv_f3
- train_source: train+valid
- valid_fold: 3
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
| 0 | 0.6943 | 0.6903 | 0.6983 | 0.7698 | 0.7558 | 0.7234 | 0.4364 | vfold505 |
| 1 | 0.6989 | 0.6947 | 0.7031 | 0.7739 | 0.7585 | 0.7261 | 0.4411 | vfold505 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
