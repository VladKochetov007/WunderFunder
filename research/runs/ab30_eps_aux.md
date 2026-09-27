# Run card: ab30_eps_aux

## Config

- name: ab30_eps_aux
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
| 0 | 0.6669 | 0.6593 | 0.6746 | 0.7752 | 0.7525 | 0.7182 | 0.44 | devB764 |
| 1 | 0.6736 | 0.6670 | 0.6802 | 0.7827 | 0.7574 | 0.7267 | 0.4482 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab30_eps_aux - ab24_eps003 lam=0.0: fold0 +0.0007±0.0007 (0.6708 vs 0.6701)  fold1 +0.0002±0.0007 (0.6575 vs 0.6572)  fold2 +0.0014±0.0005 (0.6952 vs 0.6938)  fold3 +0.0014±0.0005 (0.7004 vs 0.6990)  all +0.0010±0.0003 (0.6829 vs 0.6819)  pass=True`
- `ab30_eps_aux - ab24_eps003 lam=0.02: fold0 +0.0006±0.0007 (0.6501 vs 0.6495)  fold1 +0.0003±0.0006 (0.6358 vs 0.6355)  fold2 +0.0014±0.0005 (0.6759 vs 0.6746)  fold3 +0.0013±0.0005 (0.6808 vs 0.6795)  all +0.0009±0.0003 (0.6625 vs 0.6616)  pass=True`
