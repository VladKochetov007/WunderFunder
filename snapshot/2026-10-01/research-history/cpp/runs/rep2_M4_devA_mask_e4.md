# Run card: rep2_M4_devA_mask_e4

## Config

- name: rep2_M4_devA_mask_e4
- train_source: valid_devA
- eval_source: devB
- seed: 1
- epochs: 4
- lr: 0.0002
- wd: 0.01
- propensity: 
- loss: mix
- init: ab1_A_mix
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6249 | nan | nan | 0.7789 | 0.7522 | 0.7137 | 0.4266 | devB764 |
| 1 | 0.6339 | nan | nan | 0.7811 | 0.7549 | 0.7201 | 0.444 | devB764 |
| 2 | 0.6316 | nan | nan | 0.7809 | 0.7558 | 0.7195 | 0.4379 | devB764 |
| 3 | 0.6318 | nan | nan | 0.7812 | 0.7558 | 0.7191 | 0.4385 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
