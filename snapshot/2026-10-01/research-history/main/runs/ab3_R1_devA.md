# Run card: ab3_R1_devA

## Config

- name: ab3_R1_devA
- train_source: valid_devA
- eval_source: devB
- seed: 1
- epochs: 2
- lr: 0.0002
- wd: 0.01
- loss: mix
- init: ab1_A_mix
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5927 | nan | nan | 0.7729 | 0.7458 | 0.702 | 0.3553 | devB764 |
| 1 | 0.6012 | nan | nan | 0.7736 | 0.7465 | 0.7041 | 0.3757 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
