# Run card: ab45_lpft_f0

## Config

- name: ab45_lpft_f0
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
- init: ab44_head_f0
- freeze: 
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6715 | 0.6648 | 0.6783 | 0.802 | 0.753 | 0.7475 | 0.4378 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab45_lpft - ab29_mask lam=0.0: 0: +0.0019±0.0008 (0.6715 vs 0.6696)  1: +0.0006±0.0013 (0.6574 vs 0.6568)  pooled: +0.0013±0.0008 (0.6647 vs 0.6634)`
- `ab45_lpft - ab29_mask lam=0.02: 0: +0.0014±0.0007 (0.6500 vs 0.6486)  1: -0.0001±0.0013 (0.6345 vs 0.6346)  pooled: +0.0006±0.0007 (0.6424 vs 0.6418)`
- `ab45_lpft - ab39_ftm lam=0.0: 0: +0.0002±0.0001 (0.6715 vs 0.6713)  1: +0.0002±0.0002 (0.6574 vs 0.6572)  pooled: +0.0002±0.0001 (0.6647 vs 0.6645)`
- `ab45_lpft - ab39_ftm lam=0.02: 0: +0.0001±0.0001 (0.6500 vs 0.6499)  1: +0.0001±0.0001 (0.6345 vs 0.6344)  pooled: +0.0001±0.0001 (0.6424 vs 0.6424)`
