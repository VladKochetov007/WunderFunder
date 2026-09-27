# Run card: ab31_classw

## Config

- name: ab31_classw
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
| 0 | 0.6604 | 0.6545 | 0.6664 | 0.7756 | 0.7484 | 0.7173 | 0.4276 | devB764 |
| 1 | 0.6712 | 0.6643 | 0.6780 | 0.7814 | 0.7556 | 0.7245 | 0.445 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab31_classw - ab20_gru128_e2 lam=0.0: fold0 +0.0012±0.0006 (0.6683 vs 0.6671)  fold1 +0.0001±0.0006 (0.6553 vs 0.6552)  fold2 +0.0014±0.0004 (0.6931 vs 0.6917)  fold3 +0.0002±0.0004 (0.6981 vs 0.6980)  all +0.0007±0.0003 (0.6806 vs 0.6799)  pass=True`
- `ab31_classw - ab20_gru128_e2 lam=0.02: fold0 +0.0014±0.0006 (0.6471 vs 0.6457)  fold1 +0.0004±0.0006 (0.6330 vs 0.6325)  fold2 +0.0015±0.0004 (0.6731 vs 0.6715)  fold3 +0.0004±0.0004 (0.6780 vs 0.6776)  all +0.0009±0.0002 (0.6596 vs 0.6587)  pass=True`
- `ab31_classw - ab24_eps003 lam=0.0: fold0 -0.0018±0.0007 (0.6683 vs 0.6701)  fold1 -0.0019±0.0006 (0.6553 vs 0.6572)  fold2 -0.0008±0.0004 (0.6931 vs 0.6938)  fold3 -0.0009±0.0004 (0.6981 vs 0.6990)  all -0.0013±0.0003 (0.6806 vs 0.6819)  pass=True`
- `ab31_classw - ab24_eps003 lam=0.02: fold0 -0.0024±0.0007 (0.6471 vs 0.6495)  fold1 -0.0025±0.0006 (0.6330 vs 0.6355)  fold2 -0.0015±0.0004 (0.6731 vs 0.6746)  fold3 -0.0015±0.0004 (0.6780 vs 0.6795)  all -0.0019±0.0003 (0.6596 vs 0.6616)  pass=True`
