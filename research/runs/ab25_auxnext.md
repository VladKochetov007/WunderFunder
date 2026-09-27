# Run card: ab25_auxnext

## Config

- name: ab25_auxnext
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
| 0 | 0.6636 | 0.6564 | 0.6707 | 0.7718 | 0.7504 | 0.7164 | 0.4385 | devB764 |
| 1 | 0.6721 | 0.6653 | 0.6789 | 0.7812 | 0.7559 | 0.7238 | 0.4481 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab25_auxnext - ab20_gru128_e2 lam=0.0: fold0 +0.0025±0.0012 (0.6697 vs 0.6671)  fold1 +0.0009±0.0010 (0.6562 vs 0.6552)  fold2 +0.0010±0.0009 (0.6926 vs 0.6917)  fold3 +0.0014±0.0006 (0.6994 vs 0.6980)  all +0.0014±0.0004 (0.6814 vs 0.6799)  pass=True`
- `ab25_auxnext - ab20_gru128_e2 lam=0.02: fold0 +0.0024±0.0011 (0.6481 vs 0.6457)  fold1 +0.0009±0.0009 (0.6335 vs 0.6325)  fold2 +0.0010±0.0008 (0.6726 vs 0.6715)  fold3 +0.0013±0.0006 (0.6789 vs 0.6776)  all +0.0014±0.0004 (0.6601 vs 0.6587)  pass=True`
