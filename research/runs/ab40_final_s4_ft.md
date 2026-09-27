# Run card: ab40_final_s4_ft

## Config

- name: ab40_final_s4_ft
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
- init: ab33_final_s4
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6924 | 0.6817 | 0.7031 | 0.7976 | 0.757 | 0.7291 | 0.4447 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
