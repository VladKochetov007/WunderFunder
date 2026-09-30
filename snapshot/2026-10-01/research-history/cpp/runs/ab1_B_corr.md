# Run card: ab1_B_corr

## Config

- name: ab1_B_corr
- seed: 1
- epochs: 2
- lr: 0.001
- wd: 0.01
- loss: corr

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.2937 | nan | nan | 0.5162 | 0.3132 | 0.2026 | 0.2306 | dev600 |
| 1 | 0.2575 | nan | nan | 0.3773 | 0.2298 | 0.1973 | 0.251 | dev600 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
