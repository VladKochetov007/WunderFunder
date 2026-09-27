# Run card: ab39_ftm_f1

## Config

- name: ab39_ftm_f1
- train_source: valid
- valid_fold: 1
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
- init: ab29_mask_f1
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6572 | 0.6502 | 0.6642 | 0.7909 | 0.7542 | 0.7258 | 0.4349 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab39_ftm - ab29_mask lam=0.0: 0: +0.0017±0.0007 (0.6713 vs 0.6696)  1: +0.0004±0.0012 (0.6572 vs 0.6568)  pooled: +0.0011±0.0008 (0.6645 vs 0.6634)`
- `ab39_ftm - ab29_mask lam=0.05: 0: +0.0007±0.0006 (0.6252 vs 0.6245)  1: -0.0008±0.0012 (0.6082 vs 0.6090)  pooled: -0.0000±0.0007 (0.6170 vs 0.6170)`
