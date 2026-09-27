# Run card: ab17_ce_mse

## Config

- name: ab17_ce_mse
- train_source: train
- eval_source: devB
- seed: 3
- epochs: 1
- lr: 0.001
- wd: 0.01
- propensity: prop_v1
- loss: ce_joint
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6645 | 0.6573 | 0.6717 | 0.775 | 0.7533 | 0.7124 | 0.4345 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
