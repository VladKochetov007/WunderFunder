# Run card: ab41_fte3lr1_f0

## Config

- name: ab41_fte3lr1_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 3
- lr: 0.0001
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: ab29_mask_f0
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6706 | 0.6636 | 0.6777 | 0.8018 | 0.7524 | 0.7464 | 0.4395 | vfold468 |
| 1 | 0.6711 | 0.6640 | 0.6781 | 0.8023 | 0.753 | 0.7471 | 0.4394 | vfold468 |
| 2 | 0.6711 | 0.6641 | 0.6782 | 0.8024 | 0.753 | 0.747 | 0.4389 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab41_fte3lr1 - ab29_mask lam=0.0: 0: +0.0016±0.0009 (0.6711 vs 0.6696)  1: +0.0003±0.0011 (0.6571 vs 0.6568)  pooled: +0.0009±0.0009 (0.6643 vs 0.6634)`
- `ab41_fte3lr1 - ab29_mask lam=0.05: 0: +0.0005±0.0008 (0.6250 vs 0.6245)  1: -0.0008±0.0011 (0.6082 vs 0.6090)  pooled: -0.0001±0.0008 (0.6168 vs 0.6170)`
