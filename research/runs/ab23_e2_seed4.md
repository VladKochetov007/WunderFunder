# Run card: ab23_e2_seed4

## Config

- name: ab23_e2_seed4
- train_source: train
- eval_source: devB
- seed: 4
- epochs: 2
- lr: 0.001
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
| 0 | 0.6635 | 0.6590 | 0.6680 | 0.7767 | 0.7496 | 0.7173 | 0.436 | devB764 |
| 1 | 0.6721 | 0.6652 | 0.6789 | 0.78 | 0.7562 | 0.723 | 0.4473 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
