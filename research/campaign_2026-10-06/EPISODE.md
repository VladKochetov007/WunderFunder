# Episode loss vs row loss, linear, far no-trade cell

Matched ridge, alpha 1 frozen, wave1b split of non-holdout validation, features = wave-1 clock bundle plus the frozen blend. Eval is far rows with no trade in the last 8.

| Model | WP |
|---|---:|
| Blend | 0.404468 |
| Affine of the blend | (see `wave1_episode.json`) |
| Row-loss linear | 0.403752 |
| Episode-loss linear | 0.386754 |

Episode vs row: **−0.016997**. Kill bar was +0.005. This feature family does not pay for collapsing a stretch to one outcome.

The eval blend here is the split-bit-1 slice of the no-trade far cell, so 0.404 is not the full-cell 0.424. The comparison is the two losses, not a public forecast.

A recurrent model could still credit a constant episode differently. That fit does not run as GRU256 in 10 GB.
