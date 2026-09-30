# Run card: ab10_h128_pow2_seed4

## Config

- name: ab10_h128_pow2_seed4
- train_source: train
- eval_source: devB
- seed: 4
- epochs: 1
- lr: 0.001
- wd: 0.01
- propensity: prop_v1
- loss: mix
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6314 | nan | nan | 0.7775 | 0.7534 | 0.7182 | 0.441 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
