# Run card: ab56_ema_f1

## Config

- name: ab56_ema_f1
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
- ema_teacher: 1.0
- ema_decay: 0.999
- eval_ema: True
- loss: wmse
- init: 
- freeze: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6506 | 0.6435 | 0.6577 | 0.7852 | 0.7467 | 0.7181 | 0.4253 | vfold457 |
| 1 | 0.6579 | 0.6510 | 0.6649 | 0.791 | 0.7524 | 0.7256 | 0.4322 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab56_ema - ab46_ref lam=0.0: 0: -0.0011±0.0009 (0.6713 vs 0.6724)  1: -0.0011±0.0007 (0.6579 vs 0.6590)  pooled: -0.0011±0.0006 (0.6648 vs 0.6659)`
- `ab56_ema - ab46_ref lam=0.02: 0: -0.0012±0.0009 (0.6508 vs 0.6520)  1: -0.0010±0.0008 (0.6366 vs 0.6375)  pooled: -0.0011±0.0006 (0.6439 vs 0.6450)`
