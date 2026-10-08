# E540 fixed linear/PLE embedding comparison

Evaluated 2026-10-08T08:40:02.163719+00:00. Independently reviewed root grader, fixed epoch0 and excluded468 groups; previously exposed discovery evidence.

| PLE weight | Official WP | Delta versus linear |
|---:|---:|---:|
| 0.0 | 0.6361859352 | +0.0000000000 |
| 0.5 | 0.6349392976 | -0.0012466376 |
| 1.0 | 0.6254095463 | -0.0107763889 |

The PLE-only endpoint and predeclared equal mixture are both lower than the linear control. No exact-recipe promotion. This is a one-epoch/subset/GRU128 experiment, not a rejection of numerical embedding at other scales, as a teacher representation, or under different interactions. No public-score forecast.

[Complete machine-readable result](E540_FIXED_PAIR_RESULT_20261008_R1.json); [actual-artifact admission](E540_FIXED_PAIR_ROOT_GRADE_ADMISSION_20261008_R1.json).
