# Run card: ab34_final_distill_s4

## Config

- name: ab34_final_distill_s4
- train_source: train+valid
- valid_fold: -1
- eval_source: holdout
- seed: 4
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6883 | 0.6765 | 0.7002 | 0.7946 | 0.7492 | 0.7168 | 0.4357 | holdout368 |
| 1 | 0.6914 | 0.6793 | 0.7035 | 0.796 | 0.7533 | 0.7249 | 0.4413 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
