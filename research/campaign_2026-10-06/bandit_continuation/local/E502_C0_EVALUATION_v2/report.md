# C0 paired continuation evaluation

Metric: `global_wp_clip2_v1`. Fold 0 is exposed development data, not a fresh holdout.
The 95% intervals resample frozen evaluation components conditional on the fixed checkpoints;
they do not include training-seed variation or establish public transfer.
The comparison is continuation versus matching warm parents, not the public-best blend.
Candidate manifest validation: checkpoint CLI arguments matched to the frozen E-502 commands.

| Comparison | Candidate WP | Parent WP | Difference | 95% component interval |
| --- | ---: | ---: | ---: | ---: |
| seed3_candidate_vs_parent | 0.67995246 | 0.68094121 | -0.00098874 | [-0.00467332, +0.00244402] |
| seed4_candidate_vs_parent | 0.67997080 | 0.68102938 | -0.00105858 | [-0.00804564, +0.00374132] |
| equal_pair_candidate_vs_parent | 0.68115296 | 0.68134198 | -0.00018902 | [-0.00536403, +0.00354846] |

Evaluation sequences: 468; components: 141; selected rows: 808863.
The equal seed pair uses fixed 0.5/0.5 weights; target and prediction clipping occurs inside the official metric after that blend.
