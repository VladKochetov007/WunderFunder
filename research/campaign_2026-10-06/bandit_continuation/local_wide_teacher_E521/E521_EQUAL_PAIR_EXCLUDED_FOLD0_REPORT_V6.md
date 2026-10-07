# E521 equal-pair descriptive grade on exposed fold 0 (V6)

Official metric: `global_wp_clip2_v1`; both checkpoints are fixed at zero-based epoch 3.

| Predictor | WP | T0 | T1 |
|---|---:|---:|---:|
| seed3 | 0.67281047 | 0.66274192 | 0.68287902 |
| seed4 | 0.67127264 | 0.66205937 | 0.68048590 |
| equal_pair | 0.67409316 | 0.66452484 | 0.68366147 |

Paired component-bootstrap contrasts for the equal pair:

- pair_minus_seed3: +0.00128269; conditional 95% component interval [-0.00190647, +0.00474283].
- pair_minus_seed4: +0.00282052; conditional 95% component interval [-0.00009253, +0.00591236].

Evaluation used 468 fold-0 groups across 141 connected components and 808,863 official selected rows.

These are exposed development groups, not fresh confirmation. The resampling intervals condition on the two fitted predictors; they do not measure training-seed variability. There is no matched GRU-128 control, so this does not estimate a width effect. No public forecast is made.

Frozen grading spec SHA-256: `d6f2b76737a80f61e61a3c3a793fffced5df6e27dbd6d803fcb59bcceee45bcc`.
