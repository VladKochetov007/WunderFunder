# Run card: ab19_ft_devA

## Config

- name: ab19_ft_devA
- train_source: valid_devA
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.0002
- wd: 0.01
- propensity: 
- loss: wmse
- init: ab18_wmse_struct
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6676 | 0.6593 | 0.6758 | 0.7803 | 0.7558 | 0.7217 | 0.4432 | devB764 |
| 1 | 0.6695 | 0.6621 | 0.6768 | 0.7802 | 0.7564 | 0.7221 | 0.4447 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
