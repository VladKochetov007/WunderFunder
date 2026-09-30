# Run card: ab33_final_s4

## Config

- name: ab33_final_s4
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
| 0 | 0.6871 | 0.6761 | 0.6981 | 0.7936 | 0.75 | 0.712 | 0.4287 | holdout368 |
| 1 | 0.6916 | 0.6801 | 0.7030 | 0.7963 | 0.7541 | 0.7249 | 0.441 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
