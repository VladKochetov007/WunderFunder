# Run card: ab9_stack_devAmask

## Config

- name: ab9_stack_devAmask
- train_source: valid_devA
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.0002
- wd: 0.01
- propensity: 
- loss: mix
- init: ab6_prop_pow2_e1
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6389 | nan | nan | 0.7823 | 0.7572 | 0.7258 | 0.452 | devB764 |
| 1 | 0.6360 | nan | nan | 0.7831 | 0.7569 | 0.7225 | 0.4459 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
