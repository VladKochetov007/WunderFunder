# Run card: ab25_onlinenorm

## Config

- name: ab25_onlinenorm
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
- loss: wmse
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6628 | 0.6529 | 0.6727 | 0.7782 | 0.755 | 0.7168 | 0.4405 | devB764 |
| 1 | 0.6702 | 0.6627 | 0.6776 | 0.7805 | 0.7554 | 0.7205 | 0.4466 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab25_onlinenorm - ab20_gru128_e2 lam=0.0: fold0 -0.6671±0.0276 (0.0000 vs 0.6671)  fold1 -0.6552±0.0252 (0.0000 vs 0.6552)  fold2 -0.6917±0.0198 (0.0000 vs 0.6917)  fold3 -0.6980±0.0186 (0.0000 vs 0.6980)  all -0.6799±0.0116 (0.0000 vs 0.6799)  pass=True`
- `ab25_onlinenorm - ab20_gru128_e2 lam=0.02: fold0 -0.6457±0.0252 (0.0000 vs 0.6457)  fold1 -0.6325±0.0231 (0.0000 vs 0.6325)  fold2 -0.6715±0.0190 (0.0000 vs 0.6715)  fold3 -0.6776±0.0175 (0.0000 vs 0.6776)  all -0.6587±0.0108 (0.0000 vs 0.6587)  pass=True`
- `ab25_onlinenorm - ab20_gru128_e2 lam=0.0: fold0 +0.0021±0.0014 (0.6692 vs 0.6671)  fold1 -0.0007±0.0013 (0.6545 vs 0.6552)  fold2 +0.0003±0.0012 (0.6920 vs 0.6917)  fold3 -0.0002±0.0011 (0.6978 vs 0.6980)  all +0.0003±0.0006 (0.6803 vs 0.6799)  pass=False`
- `ab25_onlinenorm - ab20_gru128_e2 lam=0.05: fold0 +0.0019±0.0012 (0.6229 vs 0.6210)  fold1 -0.0002±0.0011 (0.6063 vs 0.6065)  fold2 +0.0002±0.0011 (0.6477 vs 0.6475)  fold3 -0.0001±0.0010 (0.6533 vs 0.6535)  all +0.0004±0.0006 (0.6343 vs 0.6339)  pass=False`
- `ab25_onlinenorm - ab20_gru128_e2 lam=0.05: fold0 +0.0017±0.0017 (0.6074 vs 0.6057)  fold1 +0.0011±0.0015 (0.5899 vs 0.5888)  fold2 -0.0008±0.0015 (0.6344 vs 0.6352)  fold3 -0.0005±0.0009 (0.6514 vs 0.6519)  all +0.0003±0.0007 (0.6237 vs 0.6235)  pass=False`
