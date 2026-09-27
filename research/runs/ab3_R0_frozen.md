# Run card: ab3_R0_frozen

## Config

- name: ab3_R0_frozen
- train_source: train
- eval_source: devB
- seed: 1
- epochs: 2
- lr: 1e-09
- wd: 0.01
- loss: mix
- init: ab1_A_mix
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5998 | nan | nan | 0.7511 | 0.7251 | 0.6985 | 0.3978 | devB764 |
| 1 | 0.5998 | nan | nan | 0.7511 | 0.7251 | 0.6985 | 0.3978 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
