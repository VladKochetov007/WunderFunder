# Run card: ab26_tv_f1

## Config

- name: ab26_tv_f1
- train_source: train+valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6493 | 0.6407 | 0.6579 | 0.7847 | 0.747 | 0.7136 | 0.4192 | vfold457 |
| 1 | 0.6559 | 0.6491 | 0.6627 | 0.7879 | 0.7498 | 0.7182 | 0.4307 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
