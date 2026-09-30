# Run card: ab3_R3_more_train

## Config

- name: ab3_R3_more_train
- train_source: train
- eval_source: devB
- seed: 7
- epochs: 2
- lr: 0.0002
- wd: 0.01
- loss: mix
- init: ab1_A_mix
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5971 | nan | nan | 0.7679 | 0.7392 | 0.6971 | 0.376 | devB764 |
| 1 | 0.6017 | nan | nan | 0.7674 | 0.7382 | 0.6958 | 0.3895 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
