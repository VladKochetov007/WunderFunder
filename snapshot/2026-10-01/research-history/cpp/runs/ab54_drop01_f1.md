# Run card: ab54_drop01_f1

## Config

- name: ab54_drop01_f1
- train_source: train+valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: 0.03
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6519 | 0.6446 | 0.6592 | 0.786 | 0.7467 | 0.7184 | 0.4252 | vfold457 |
| 1 | 0.6591 | 0.6523 | 0.6658 | 0.7913 | 0.7526 | 0.7264 | 0.4336 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab54_drop01 - ab46_ref lam=0.0: 0: -0.0003±0.0006 (0.6721 vs 0.6724)  1: +0.0000±0.0004 (0.6591 vs 0.6590)  pooled: -0.0001±0.0003 (0.6658 vs 0.6659)`
- `ab54_drop01 - ab46_ref lam=0.02: 0: -0.0004±0.0005 (0.6516 vs 0.6520)  1: +0.0000±0.0004 (0.6376 vs 0.6375)  pooled: -0.0002±0.0003 (0.6448 vs 0.6450)`
