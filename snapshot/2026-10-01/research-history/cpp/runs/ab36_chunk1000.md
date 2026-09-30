# Run card: ab36_chunk1000

## Config

- name: ab36_chunk1000
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
| 0 | 0.6683 | 0.6610 | 0.6756 | 0.7774 | 0.7535 | 0.7219 | 0.4421 | devB764 |
| 1 | 0.6742 | 0.6677 | 0.6808 | 0.784 | 0.7586 | 0.7281 | 0.4504 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab36_chunk1000 - ab30_eps_aux lam=0.0: fold0 +0.0004±0.0018 (0.6712 vs 0.6708)  fold1 +0.0007±0.0006 (0.6582 vs 0.6575)  fold2 +0.0001±0.0005 (0.6953 vs 0.6952)  fold3 +0.0007±0.0006 (0.7011 vs 0.7004)  all +0.0005±0.0005 (0.6833 vs 0.6829)  pass=False`
- `ab36_chunk1000 - ab30_eps_aux lam=0.05: fold0 +0.0005±0.0016 (0.6267 vs 0.6263)  fold1 +0.0005±0.0005 (0.6115 vs 0.6110)  fold2 +0.0002±0.0005 (0.6531 vs 0.6529)  fold3 +0.0007±0.0005 (0.6584 vs 0.6577)  all +0.0005±0.0004 (0.6391 vs 0.6387)  pass=False`
- `ab36_chunk1000 - ab30_eps_aux lam=0.05: fold0 +0.0011±0.0010 (0.6113 vs 0.6102)  fold1 +0.0006±0.0006 (0.5947 vs 0.5941)  fold2 -0.0001±0.0008 (0.6406 vs 0.6407)  fold3 +0.0010±0.0007 (0.6564 vs 0.6554)  all +0.0007±0.0004 (0.6287 vs 0.6280)  pass=False`
