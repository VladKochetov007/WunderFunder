# Run card: ab36_chunk500

## Config

- name: ab36_chunk500
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
| 0 | 0.6688 | 0.6607 | 0.6770 | 0.7803 | 0.7504 | 0.7169 | 0.4456 | devB764 |
| 1 | 0.6746 | 0.6677 | 0.6815 | 0.7841 | 0.7587 | 0.7279 | 0.454 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab36_chunk500 - ab30_eps_aux lam=0.0: fold0 -0.0017±0.0025 (0.6691 vs 0.6708)  fold1 +0.0007±0.0009 (0.6581 vs 0.6575)  fold2 +0.0020±0.0007 (0.6972 vs 0.6952)  fold3 +0.0011±0.0007 (0.7015 vs 0.7004)  all +0.0006±0.0007 (0.6835 vs 0.6829)  pass=False`
- `ab36_chunk500 - ab30_eps_aux lam=0.05: fold0 -0.0013±0.0023 (0.6250 vs 0.6263)  fold1 +0.0005±0.0008 (0.6115 vs 0.6110)  fold2 +0.0020±0.0006 (0.6549 vs 0.6529)  fold3 +0.0012±0.0006 (0.6589 vs 0.6577)  all +0.0007±0.0007 (0.6393 vs 0.6387)  pass=False`
- `ab36_chunk500 - ab30_eps_aux lam=0.05: fold0 +0.0009±0.0015 (0.6111 vs 0.6102)  fold1 +0.0014±0.0011 (0.5955 vs 0.5941)  fold2 +0.0022±0.0008 (0.6429 vs 0.6407)  fold3 +0.0017±0.0007 (0.6571 vs 0.6554)  all +0.0016±0.0004 (0.6296 vs 0.6280)  pass=True`
