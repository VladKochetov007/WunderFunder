# Run card: rep1_R1_devA_seed2

## Config

- name: rep1_R1_devA_seed2
- train_source: valid_devA
- eval_source: devB
- seed: 2
- epochs: 2
- lr: 0.0002
- wd: 0.01
- propensity: 
- loss: mix
- init: ab1_A_mix
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6008 | nan | nan | 0.7719 | 0.7456 | 0.7036 | 0.3765 | devB764 |
| 1 | 0.5998 | nan | nan | 0.7732 | 0.7462 | 0.6985 | 0.376 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
