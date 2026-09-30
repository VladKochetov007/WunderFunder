# Run card: ab23_e2_lr2e3

## Config

- name: ab23_e2_lr2e3
- train_source: train
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.002
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6604 | 0.6554 | 0.6654 | 0.7733 | 0.7451 | 0.7139 | 0.4311 | devB764 |
| 1 | 0.6712 | 0.6640 | 0.6785 | 0.781 | 0.754 | 0.7203 | 0.4474 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
