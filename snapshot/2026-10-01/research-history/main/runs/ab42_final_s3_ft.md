# Run card: ab42_final_s3_ft

## Config

- name: ab42_final_s3_ft
- train_source: valid
- valid_fold: -1
- eval_source: holdout
- seed: 3
- epochs: 1
- lr: 0.0002
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: ab33_final_s3
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6915 | 0.6799 | 0.7032 | 0.7975 | 0.7573 | 0.728 | 0.4438 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
