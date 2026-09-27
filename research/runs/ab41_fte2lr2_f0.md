# Run card: ab41_fte2lr2_f0

## Config

- name: ab41_fte2lr2_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 2
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
| 0 | 0.6707 | 0.6637 | 0.6776 | 0.802 | 0.7533 | 0.7466 | 0.4398 | vfold468 |
| 1 | 0.6715 | 0.6645 | 0.6784 | 0.8025 | 0.7537 | 0.7474 | 0.4391 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab41_fte2lr2 - ab29_mask lam=0.0: 0: +0.0019±0.0010 (0.6715 vs 0.6696)  1: +0.0008±0.0009 (0.6576 vs 0.6568)  pooled: +0.0013±0.0008 (0.6647 vs 0.6634)`
- `ab41_fte2lr2 - ab29_mask lam=0.05: 0: +0.0005±0.0009 (0.6250 vs 0.6245)  1: -0.0002±0.0009 (0.6088 vs 0.6090)  pooled: +0.0001±0.0007 (0.6171 vs 0.6170)`
