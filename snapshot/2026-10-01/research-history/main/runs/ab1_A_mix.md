# Run card: ab1_A_mix

## Config

- name: ab1_A_mix
- seed: 1
- epochs: 2
- lr: 0.001
- wd: 0.01
- loss: mix

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6036 | nan | nan | 0.7575 | 0.7271 | 0.6915 | 0.4002 | dev600 |
| 1 | 0.5983 | nan | nan | 0.7769 | 0.7438 | 0.686 | 0.3708 | dev600 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
