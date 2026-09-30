# Run card: ab41_fte3lr1_f1

## Config

- name: ab41_fte3lr1_f1
- train_source: valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 3
- lr: 0.0001
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
| 0 | 0.6570 | 0.6506 | 0.6634 | 0.7903 | 0.7541 | 0.7252 | 0.4347 | vfold457 |
| 1 | 0.6568 | 0.6499 | 0.6636 | 0.7909 | 0.7542 | 0.7261 | 0.4347 | vfold457 |
| 2 | 0.6571 | 0.6501 | 0.6641 | 0.7909 | 0.7544 | 0.7262 | 0.4349 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab41_fte3lr1 - ab29_mask lam=0.0: 0: +0.0016±0.0009 (0.6711 vs 0.6696)  1: +0.0003±0.0011 (0.6571 vs 0.6568)  pooled: +0.0009±0.0009 (0.6643 vs 0.6634)`
- `ab41_fte3lr1 - ab29_mask lam=0.05: 0: +0.0005±0.0008 (0.6250 vs 0.6245)  1: -0.0008±0.0011 (0.6082 vs 0.6090)  pooled: -0.0001±0.0008 (0.6168 vs 0.6170)`
