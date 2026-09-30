# Run card: ab39_ftm_f0

## Config

- name: ab39_ftm_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 1
- lr: 0.0002
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
| 0 | 0.6713 | 0.6646 | 0.6781 | 0.8017 | 0.7526 | 0.7469 | 0.4375 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab39_ftm - ab29_mask lam=0.0: 0: +0.0017±0.0007 (0.6713 vs 0.6696)  1: +0.0004±0.0012 (0.6572 vs 0.6568)  pooled: +0.0011±0.0008 (0.6645 vs 0.6634)`
- `ab39_ftm - ab29_mask lam=0.05: 0: +0.0007±0.0006 (0.6252 vs 0.6245)  1: -0.0008±0.0012 (0.6082 vs 0.6090)  pooled: -0.0000±0.0007 (0.6170 vs 0.6170)`
