# Run card: ab37_ft_f1

## Config

- name: ab37_ft_f1
- train_source: valid
- valid_fold: 1
- eval_source: vfold
- seed: 3
- epochs: 1
- lr: 0.0002
- wd: 0.01
- cell: gru
- propensity: 
- prop_eps: 0.0
- valid_true_mask: -1.0
- loss: wmse
- init: ab26_tv_f1
- mask_weight: True

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6572 | 0.6503 | 0.6642 | 0.7904 | 0.7537 | 0.7234 | 0.435 | vfold457 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- `ab37_ft - ab26_tv lam=0.0: 0: +0.0019±0.0006 (0.6711 vs 0.6692)  1: +0.0014±0.0011 (0.6572 vs 0.6559)  pooled: +0.0016±0.0006 (0.6644 vs 0.6628)`
- `ab37_ft - ab26_tv lam=0.02: 0: +0.0015±0.0005 (0.6493 vs 0.6478)  1: +0.0010±0.0011 (0.6343 vs 0.6333)  pooled: +0.0012±0.0006 (0.6420 vs 0.6408)`
