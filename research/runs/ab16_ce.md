# Run card: ab16_ce

## Config

- name: ab16_ce
- train_source: train
- eval_source: devB
- seed: 3
- epochs: 1
- lr: 0.001
- wd: 0.01
- propensity: prop_v1
- loss: ce_joint
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6489 | 0.6408 | 0.6570 | 0.7597 | 0.7293 | 0.6758 | 0.4189 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
