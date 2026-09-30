# Run card: ab31_classw_eps

## Config

- name: ab31_classw_eps
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
| 0 | 0.6588 | 0.6525 | 0.6651 | 0.7758 | 0.7504 | 0.7255 | 0.4274 | devB764 |
| 1 | 0.6726 | 0.6658 | 0.6793 | 0.7825 | 0.7577 | 0.7286 | 0.4443 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab31_classw_eps - ab20_gru128_e2 lam=0.0: fold0 +0.0021±0.0010 (0.6692 vs 0.6671)  fold1 +0.0019±0.0009 (0.6572 vs 0.6552)  fold2 +0.0027±0.0007 (0.6944 vs 0.6917)  fold3 +0.0015±0.0005 (0.6994 vs 0.6980)  all +0.0020±0.0004 (0.6820 vs 0.6799)  pass=True`
- `ab31_classw_eps - ab20_gru128_e2 lam=0.02: fold0 +0.0032±0.0010 (0.6489 vs 0.6457)  fold1 +0.0031±0.0008 (0.6356 vs 0.6325)  fold2 +0.0036±0.0007 (0.6752 vs 0.6715)  fold3 +0.0024±0.0006 (0.6800 vs 0.6776)  all +0.0031±0.0004 (0.6617 vs 0.6587)  pass=True`
- `ab31_classw_eps - ab24_eps003 lam=0.0: fold0 -0.0009±0.0011 (0.6692 vs 0.6701)  fold1 -0.0000±0.0005 (0.6572 vs 0.6572)  fold2 +0.0005±0.0005 (0.6944 vs 0.6938)  fold3 +0.0004±0.0005 (0.6994 vs 0.6990)  all +0.0000±0.0003 (0.6820 vs 0.6819)  pass=False`
- `ab31_classw_eps - ab24_eps003 lam=0.02: fold0 -0.0006±0.0010 (0.6489 vs 0.6495)  fold1 +0.0001±0.0004 (0.6356 vs 0.6355)  fold2 +0.0006±0.0005 (0.6752 vs 0.6746)  fold3 +0.0005±0.0005 (0.6800 vs 0.6795)  all +0.0002±0.0003 (0.6617 vs 0.6616)  pass=False`
