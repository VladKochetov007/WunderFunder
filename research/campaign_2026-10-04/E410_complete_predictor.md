# E410: complete predictor comparison

The frozen equal parent/head combo gains **+0.002414 WP** over the incumbent's
same procedure across four component-excluded, historically exposed folds.
The paired component-bootstrap 95% interval is **[+0.001566, +0.003449]**,
conditional on the fitted seed3 models. It does not include training-seed or
research-selection uncertainty. There is no new public result.

| Fold | Delta WP | Component 95% interval |
|---|---:|---:|
| 0 | +0.004613 | [+0.001687, +0.007846] |
| 1 | +0.000811 | [-0.000752, +0.002292] |
| 2 | +0.000983 | [-0.000148, +0.002054] |
| 3 | +0.002992 | [+0.001779, +0.004576] |

Pooled CTRL WP is 0.691244; COMBO is 0.693658. Pooled class3 improves
0.437950 to 0.442225, but folds1 and2 class3 each decline about 0.00056.
Global correlation is not a linear weighted average of class correlations.

The four new heads use the predeclared one-epoch historical procedure, same
component exclusions, fixed epoch0 and equal parent/head mixture. No mixture
or checkpoint search took place. The unusually large fold0 result prompted a
fresh CTRL head reproduction: every saved model tensor exactly equals the
historical CTRL head. This supports transport of the head-training procedure.

All four head fits and prediction exports completed. Observed head-fit peak
GPU memory was 2,039,480,320 bytes; no OOM or killed worker occurred. The runner
preserves limits, exact inputs/source hashes, commands and per-attempt logs.

Evidence: `specs/combo_merged_screen.json`,
`checks/combo_merged_all_folds_report.json`, `runner/attempts/e410-*.json`.
The live marimo notebook exports plots and their source data under
`eda/complete_predictor/`. E412 carries the fixed recipe into full-data fitting.
