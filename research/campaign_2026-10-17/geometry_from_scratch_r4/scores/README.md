# R4 official score receipts

This folder stores a separate `global_wp_clip2_v1` score receipt for every
fixed epoch-2 endpoint and the paired analysis for all completed seed/fold
pairs. The runner verifies the prediction, split and metric identities before
marking a run scored.

The primary comparison is the preregistered equal-weight mean of fold-level
paired feature-minus-control deltas. Training-seed variation and component
cluster uncertainty are reported separately. Folds are exposed or potentially
exposed, so these results can guide local development and follow-up allocation
but cannot establish a fresh holdout result or public score.
