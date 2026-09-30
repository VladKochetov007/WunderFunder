# Run card: ab28_distill3

## Config

- name: ab28_distill3
- train_source: train
- valid_fold: -1
- eval_source: devB
- seed: 3
- epochs: 2
- lr: 0.001
- wd: 0.01
- cell: gru
- propensity: prop_v1
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6639 | 0.6594 | 0.6684 | 0.7775 | 0.7518 | 0.7202 | 0.4381 | devB764 |
| 1 | 0.6718 | 0.6648 | 0.6789 | 0.7814 | 0.7565 | 0.7236 | 0.4467 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab28_distill3 - ab20_gru128_e2 lam=0.0: fold0 +0.0013±0.0008 (0.6684 vs 0.6671)  fold1 +0.0013±0.0005 (0.6565 vs 0.6552)  fold2 +0.0015±0.0004 (0.6931 vs 0.6917)  fold3 +0.0013±0.0003 (0.6992 vs 0.6980)  all +0.0013±0.0003 (0.6812 vs 0.6799)  pass=True`
- `ab28_distill3 - ab20_gru128_e2 lam=0.02: fold0 +0.0014±0.0007 (0.6471 vs 0.6457)  fold1 +0.0016±0.0005 (0.6341 vs 0.6325)  fold2 +0.0017±0.0004 (0.6732 vs 0.6715)  fold3 +0.0015±0.0003 (0.6791 vs 0.6776)  all +0.0015±0.0002 (0.6602 vs 0.6587)  pass=True`
