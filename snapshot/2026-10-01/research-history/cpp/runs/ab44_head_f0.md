# Run card: ab44_head_f0

## Config

- name: ab44_head_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 1
- lr: 0.0005
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: ab29_mask_f0
- freeze: inp,gru
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6708 | 0.6639 | 0.6777 | 0.8013 | 0.7509 | 0.7461 | 0.4372 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab44_head - ab29_mask lam=0.0: 0: +0.0012±0.0004 (0.6708 vs 0.6696)  1: +0.0005±0.0011 (0.6574 vs 0.6568)  pooled: +0.0009±0.0006 (0.6643 vs 0.6634)`
- `ab44_head - ab29_mask lam=0.02: 0: +0.0009±0.0004 (0.6495 vs 0.6486)  1: +0.0000±0.0010 (0.6346 vs 0.6346)  pooled: +0.0005±0.0006 (0.6423 vs 0.6418)`
