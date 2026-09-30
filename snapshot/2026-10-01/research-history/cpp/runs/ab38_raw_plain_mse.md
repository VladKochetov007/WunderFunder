# Run card: ab38_raw_plain_mse

## Config

- name: ab38_raw_plain_mse
- train_source: train
- valid_fold: -1
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: mse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6476 | 0.6461 | 0.6490 | 0.7576 | 0.7382 | 0.7063 | 0.4141 | devB764 |
| 1 | 0.6544 | 0.6477 | 0.6611 | 0.7642 | 0.7417 | 0.7138 | 0.4253 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
