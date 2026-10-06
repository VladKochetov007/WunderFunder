# Tiny GRU episode loss vs row loss

Hidden 32, one layer, 250 fit / 250 eval non-holdout valid sequences, wave1b split, seed 0, CPU, full-sequence BPTT, 10 GB cap. Loss is applied only on far no-trade rows. Kill bar was **+0.005 WP** episode minus matched row-loss on that cell.

| Arm | Far no-trade WP | All scored WP |
|---|---:|---:|
| Row WMSE | 0.211144 | 0.282234 |
| Episode WMSE | 0.222179 | 0.311898 |
| Delta | **+0.011035** | +0.029664 |

It clears the bar at this scale. Absolute WP is far below the champion’s 0.424 on the same cell. Linear episode collapse on the same cell was **−0.017**, so the gain is the recurrent credit through a constant episode, not the feature bundle.

Seed 1, same recipe:

| Arm | Far no-trade WP | Delta |
|---|---:|---:|
| Row WMSE | 0.206981 |  |
| Episode WMSE | 0.223924 | **+0.016943** |

Both seeds clear +0.005 when the loss is **only** on far no-trade rows. Mean protocol delta about **+0.014**.

Seed 0, same model, loss on **all scored rows**, still evaluated on the hole (`wave1_tiny_gru_all_s0.json`):

| Arm | Far no-trade WP | All scored WP |
|---|---:|---:|
| Row WMSE | 0.286498 | 0.475639 |
| Episode WMSE | 0.239161 | 0.465092 |
| Delta | **−0.047338** | −0.010547 |

A global episode objective hurts the hole.

Mixed loss (row WMSE on non-far rows + episode on far no-trade), same seed, vs row-on-all (`wave1_tiny_gru_mixed_s0.json`):

| Arm | Far no-trade WP | All scored WP |
|---|---:|---:|
| Row on all scored | 0.286498 | 0.475639 |
| Mixed | 0.263025 | 0.394608 |
| Delta | **−0.023473** | −0.081031 |

Adding episode credit on the hole while still training the rest with row loss also loses. The +0.011 / +0.017 result is two specialists trained only on the hole. It is not a drop-in extra term on a model that also has to fit the easy rows. Absolute WP stays well below the champion. Not a public increment on 0.6660.
