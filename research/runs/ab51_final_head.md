# Run card: ab51_final_head

## Config

- name: ab51_final_head
- train_source: valid
- valid_fold: -1
- eval_source: holdout
- seed: 3
- epochs: 1
- lr: 0.0005
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- ema_teacher: 0.0
- ema_decay: 0.999
- eval_ema: False
- loss: wmse
- init: ab33_final_s4
- freeze: inp,gru
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6920 | 0.6813 | 0.7026 | 0.7975 | 0.7557 | 0.7271 | 0.4436 | holdout368 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
