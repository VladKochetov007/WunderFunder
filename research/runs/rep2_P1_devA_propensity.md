# Run card: rep2_P1_devA_propensity

## Config

- name: rep2_P1_devA_propensity
- train_source: valid_devA
- eval_source: devB
- seed: 1
- epochs: 2
- lr: 0.0002
- wd: 0.01
- propensity: prop_v1
- loss: mix
- init: ab1_A_mix
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6267 | nan | nan | 0.7773 | 0.7499 | 0.7182 | 0.4305 | devB764 |
| 1 | 0.6215 | nan | nan | 0.7787 | 0.7509 | 0.7139 | 0.4183 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
