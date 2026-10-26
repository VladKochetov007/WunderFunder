<!-- tabdata-compact-factor-sweep-20261026 -->
Campaign status: **stopped_by_user**. Complete new validation points: **0/40**; 20 retained PR43 points reused for control/joint-treatment plots.

Eight new five-epoch fits; four two-B300 groups per wave. Batch128/rank, global256, 1150updates/fit. BF16/BF16x9, FP32 masters/states, warmup5% + cosine.

| Wave | Runtime cell | Scientific arm | Epochs | Update |
|---|---|---|---:|---:|
| regularization_factors | ctl_f0 | dropout_only | 0/5 | 0/1150 |
| regularization_factors | reg_f0 | decay_only | 0/5 | 0/1150 |
| regularization_factors | ctl_f2 | dropout_only | 0/5 | 0/1150 |
| regularization_factors | reg_f2 | decay_only | 0/5 | 0/1150 |
| learning_rate | ctl_f0 | lr_low | 0/5 | 0/1150 |
| learning_rate | reg_f0 | lr_high | 0/5 | 0/1150 |
| learning_rate | ctl_f2 | lr_low | 0/5 | 0/1150 |
| learning_rate | reg_f2 | lr_high | 0/5 | 0/1150 |

| Arm | Selected common epoch | F0 WP | F2 WP | Mean WP | Paired mean delta |
|---|---:|---:|---:|---:|---:|
| control | 4 | 0.6979889 | 0.6589634 | 0.6784761 | +0.0000000 |
| joint | 4 | 0.6974817 | 0.6584413 | 0.6779615 | -0.0005146 |

Train loss in figures is the same-weight model.eval fixed64 fitting probe; validation loss uses the full inner holdout, both all rows>=99. Different populations: absolute gap is not an unbiased generalization-gap estimate. Validation WP uses its official scored mask. Each arm selects a common epoch by mean F0/F2 WP, earliest exact tie. One seed per fold; all splits are development-exposed. No significance, asymptotic or public-score claim.

![Regularization train/val](https://github.com/VladKochetov007/WunderFunder/blob/experiment/compact-factor-sweep-20261026/research/experiment_drafts/compact-factor-sweep-20261026/plots/regularization.png?raw=true)

![Learning-rate train/val](https://github.com/VladKochetov007/WunderFunder/blob/experiment/compact-factor-sweep-20261026/research/experiment_drafts/compact-factor-sweep-20261026/plots/learning_rate.png?raw=true)

[Coordinates and hashes](https://github.com/VladKochetov007/WunderFunder/blob/experiment/compact-factor-sweep-20261026/research/experiment_drafts/compact-factor-sweep-20261026/plots/coordinates.json) · [CSV](https://github.com/VladKochetov007/WunderFunder/blob/experiment/compact-factor-sweep-20261026/research/experiment_drafts/compact-factor-sweep-20261026/plots/coordinates.csv) · [Selected endpoints](https://github.com/VladKochetov007/WunderFunder/blob/experiment/compact-factor-sweep-20261026/research/experiment_drafts/compact-factor-sweep-20261026/plots/selection.json)
