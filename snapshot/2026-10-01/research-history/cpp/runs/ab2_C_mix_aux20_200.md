# Run card: ab2_C_mix_aux20_200

## Config

- name: ab2_C_mix_aux20_200
- seed: 1
- epochs: 2
- lr: 0.001
- wd: 0.01
- loss: mix

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.5189 | nan | nan | 0.7689 | 0.7365 | 0.4229 | 0.3419 | dev600 |
| 1 | 0.5117 | nan | nan | 0.7704 | 0.7306 | 0.3891 | 0.3476 | dev600 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
