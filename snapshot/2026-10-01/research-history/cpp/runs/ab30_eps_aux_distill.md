# Run card: ab30_eps_aux_distill

## Config

- name: ab30_eps_aux_distill
- train_source: train
- valid_fold: -1
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.03
- valid_true_mask: -1.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6706 | 0.6634 | 0.6778 | 0.7788 | 0.7557 | 0.7232 | 0.4436 | devB764 |
| 1 | 0.6742 | 0.6674 | 0.6811 | 0.7833 | 0.7587 | 0.7277 | 0.4492 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab30_eps_aux_distill - ab24_eps003 lam=0.0: fold0 -0.0007±0.0021 (0.6694 vs 0.6701)  fold1 +0.0015±0.0005 (0.6587 vs 0.6572)  fold2 +0.0013±0.0005 (0.6951 vs 0.6938)  fold3 +0.0025±0.0005 (0.7016 vs 0.6990)  all +0.0012±0.0006 (0.6832 vs 0.6819)  pass=False`
- `ab30_eps_aux_distill - ab24_eps003 lam=0.02: fold0 -0.0007±0.0020 (0.6488 vs 0.6495)  fold1 +0.0016±0.0005 (0.6372 vs 0.6355)  fold2 +0.0012±0.0005 (0.6758 vs 0.6746)  fold3 +0.0025±0.0005 (0.6819 vs 0.6795)  all +0.0012±0.0005 (0.6628 vs 0.6616)  pass=False`
