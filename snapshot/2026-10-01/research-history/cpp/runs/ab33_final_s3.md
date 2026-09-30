# Run card: ab33_final_s3

## Config

- name: ab33_final_s3
- train_source: train+valid
- valid_fold: -1
- eval_source: holdout
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
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6866 | 0.6739 | 0.6993 | 0.7929 | 0.7519 | 0.7229 | 0.4343 | holdout368 |
| 1 | 0.6906 | 0.6782 | 0.7029 | 0.7947 | 0.754 | 0.7234 | 0.4422 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
