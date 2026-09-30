# Run card: ab2_E_mix_h384

## Config

- name: ab2_E_mix_h384
- train_source: train
- eval_source: dev
- seed: 1
- epochs: 2
- lr: 0.001
- wd: 0.01
- loss: mix
- init: 
- mask_weight: False

## Per-epoch validation (global WP, lambda 0; by-class per-sequence WP)

| epoch | WP | t0 | t1 | class 0 | class 1 | class 2 | class 3 | split |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.6096 | nan | nan | 0.7702 | 0.725 | 0.6857 | 0.4151 | dev600 |
| 1 | 0.5943 | nan | nan | 0.7762 | 0.7409 | 0.6698 | 0.3729 | dev600 |

Snapshots: model.pt only (run predates per-epoch snapshots)

## Fold verdicts (from queue logs)

- none yet
