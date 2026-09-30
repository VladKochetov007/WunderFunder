# Run card: ab28_distill_base

## Config

- name: ab28_distill_base
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
| 0 | 0.6617 | 0.6586 | 0.6647 | 0.7767 | 0.75 | 0.7183 | 0.436 | devB764 |
| 1 | 0.6709 | 0.6639 | 0.6780 | 0.7816 | 0.7558 | 0.7224 | 0.4461 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab28_distill_base - ab20_gru128_e2 lam=0.0: fold0 +0.0006±0.0013 (0.6677 vs 0.6671)  fold1 +0.0002±0.0008 (0.6555 vs 0.6552)  fold2 -0.0001±0.0009 (0.6916 vs 0.6917)  fold3 +0.0009±0.0007 (0.6988 vs 0.6980)  all +0.0004±0.0004 (0.6803 vs 0.6799)  pass=False`
- `ab28_distill_base - ab20_gru128_e2 lam=0.02: fold0 +0.0010±0.0012 (0.6467 vs 0.6457)  fold1 +0.0010±0.0007 (0.6335 vs 0.6325)  fold2 +0.0003±0.0008 (0.6719 vs 0.6715)  fold3 +0.0013±0.0006 (0.6788 vs 0.6776)  all +0.0009±0.0004 (0.6596 vs 0.6587)  pass=True`
