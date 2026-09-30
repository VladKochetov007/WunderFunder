# Run card: ab3_R2_devA_mask

## Config

- name: ab3_R2_devA_mask
- train_source: valid_devA
- eval_source: devB
- seed: 1
- epochs: 2
- lr: 0.0002
- wd: 0.01
- loss: mix
- init: ab1_A_mix
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6278 | nan | nan | 0.7787 | 0.7524 | 0.7161 | 0.4328 | devB764 |
| 1 | 0.6306 | nan | nan | 0.7795 | 0.7543 | 0.7191 | 0.437 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
