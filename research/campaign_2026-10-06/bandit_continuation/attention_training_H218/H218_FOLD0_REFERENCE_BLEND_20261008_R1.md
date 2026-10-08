# H218 fold-0 reference blends

This conditional comparison covers the same 468 exposed fold-0 groups for H218 and the own seed-3 GRU reference. The candidate contains only those 468 groups; its rows are joined to the reference's 1,873-row valid export and the target cache by `(valid, row_group)`, not by shared array position.

Reference baseline: WP 0.680941207.

| H218 weight | WP | t0 | t1 | delta vs reference | paired component 95% CI |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.680941207 | 0.674463034 | 0.687419380 | +0.000000000 | [+0.000000000, +0.000000000] |
| 0.10 | 0.680369396 | 0.673719239 | 0.687019554 | -0.000571811 | [-0.000871576, -0.000278931] |
| 0.25 | 0.679003381 | 0.672071432 | 0.685935330 | -0.001937826 | [-0.002709184, -0.001195943] |
| 0.50 | 0.675315228 | 0.667850858 | 0.682779597 | -0.005625979 | [-0.007252372, -0.004106785] |
| 1.00 | 0.662403646 | 0.653665596 | 0.671141696 | -0.018537561 | [-0.022156867, -0.015263852] |

Pair prediction relationship on the official scored/prediction mask and clipped-target magnitude weights:

- t0: weighted covariance 0.801775717; weighted correlation 0.978432936
- t1: weighted covariance 0.849191721; weighted correlation 0.980468081

All fixed weights are reported without selecting a winner. The interval is conditional on fixed predictions and exposed fold-0 components; it does not estimate training-seed or public-transfer uncertainty. This is not a public forecast or all-data comparison.
