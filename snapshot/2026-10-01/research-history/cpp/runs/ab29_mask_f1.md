# Run card: ab29_mask_f1

## Config

- name: ab29_mask_f1
- train_source: train+valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- valid_true_mask: 0.03
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6508 | 0.6419 | 0.6596 | 0.7856 | 0.7476 | 0.7178 | 0.4226 | vfold457 |
| 1 | 0.6568 | 0.6497 | 0.6639 | 0.7893 | 0.7508 | 0.722 | 0.4313 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab29_mask - ab26_tv lam=0.0: 0: +0.0004±0.0005 (0.6696 vs 0.6692)  1: +0.0009±0.0005 (0.6568 vs 0.6559)  pooled: +0.0006±0.0004 (0.6634 vs 0.6628)`
- `ab29_mask - ab26_tv lam=0.02: 0: +0.0008±0.0004 (0.6486 vs 0.6478)  1: +0.0013±0.0005 (0.6346 vs 0.6333)  pooled: +0.0010±0.0004 (0.6418 vs 0.6408)`
