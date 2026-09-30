# Run card: ab48_basehead_f0

## Config

- name: ab48_basehead_f0
- train_source: valid
- valid_fold: 0
- eval_source: vfold
- seed: 3
- epochs: 1
- lr: 0.0005
- wd: 0.0
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- init: base_rnn
- freeze: inp,gru,head.0
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6617 | 0.6502 | 0.6733 | 0.7891 | 0.7461 | 0.7356 | 0.433 | vfold468 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
