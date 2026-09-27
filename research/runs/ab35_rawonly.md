# Run card: ab35_rawonly

## Config

- name: ab35_rawonly
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
| 0 | 0.6658 | 0.6578 | 0.6738 | 0.7746 | 0.7514 | 0.7159 | 0.4353 | devB764 |
| 1 | 0.6719 | 0.6650 | 0.6789 | 0.7821 | 0.756 | 0.725 | 0.4443 | devB764 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab35_rawonly - ab30_eps_aux lam=0.0: fold0 -0.0018±0.0013 (0.6445 vs 0.6463)  fold1 -0.0006±0.0014 (0.6346 vs 0.6352)  fold2 -0.0033±0.0007 (0.6757 vs 0.6791)  fold3 -0.0004±0.0008 (0.6935 vs 0.6939)  all -0.0015±0.0005 (0.6654 vs 0.6669)  pass=True`
- `ab35_rawonly - ab30_eps_aux lam=0.05: fold0 -0.0024±0.0012 (0.6078 vs 0.6102)  fold1 -0.0012±0.0012 (0.5929 vs 0.5941)  fold2 -0.0038±0.0007 (0.6369 vs 0.6407)  fold3 -0.0009±0.0007 (0.6545 vs 0.6554)  all -0.0020±0.0004 (0.6260 vs 0.6280)  pass=True`
