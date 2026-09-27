# Run card: ab41_fte1lr5_f1

## Config

- name: ab41_fte1lr5_f1
- train_source: valid
- valid_fold: 1
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
- init: ab29_mask_f1
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6573 | 0.6504 | 0.6641 | 0.7912 | 0.7551 | 0.7266 | 0.4349 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab41_fte1lr5 - ab29_mask lam=0.0: 0: +0.0019±0.0010 (0.6715 vs 0.6696)  1: +0.0005±0.0017 (0.6573 vs 0.6568)  pooled: +0.0012±0.0010 (0.6646 vs 0.6634)`
- `ab41_fte1lr5 - ab29_mask lam=0.05: 0: +0.0001±0.0009 (0.6245 vs 0.6245)  1: -0.0014±0.0016 (0.6076 vs 0.6090)  pooled: -0.0006±0.0009 (0.6164 vs 0.6170)`
