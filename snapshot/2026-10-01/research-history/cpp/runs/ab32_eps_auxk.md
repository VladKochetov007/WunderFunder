# Run card: ab32_eps_auxk

## Config

- name: ab32_eps_auxk
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
| 0 | 0.6675 | 0.6615 | 0.6735 | 0.78 | 0.7553 | 0.7181 | 0.4356 | devB764 |
| 1 | 0.6727 | 0.6662 | 0.6791 | 0.783 | 0.7573 | 0.726 | 0.4462 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab32_eps_auxk - ab30_eps_aux lam=0.0: fold0 +0.0007±0.0007 (0.6715 vs 0.6708)  fold1 +0.0003±0.0008 (0.6577 vs 0.6575)  fold2 -0.0008±0.0005 (0.6944 vs 0.6952)  fold3 -0.0004±0.0006 (0.7000 vs 0.7004)  all -0.0001±0.0003 (0.6828 vs 0.6829)  pass=False`
- `ab32_eps_auxk - ab30_eps_aux lam=0.02: fold0 +0.0008±0.0006 (0.6509 vs 0.6501)  fold1 +0.0002±0.0007 (0.6360 vs 0.6358)  fold2 -0.0007±0.0005 (0.6752 vs 0.6759)  fold3 -0.0004±0.0005 (0.6804 vs 0.6808)  all -0.0000±0.0003 (0.6625 vs 0.6625)  pass=False`
