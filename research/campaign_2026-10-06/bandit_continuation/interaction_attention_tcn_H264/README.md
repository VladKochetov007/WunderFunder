# Fixed attention/TCN interaction

Both completed parent models had weaker standalone and reference-blend results. This bounded experiment asks whether a fixed equal mixture nevertheless improves the endpoint. No fitting or weight search was performed.

| Weight on equal mixture | Official WP | Delta versus reference |
| --- | ---: | ---: |
| 0 | 0.680941207 | +0.000000000 |
| 0.1 | 0.680622008 | -0.000319199 |
| 0.25 | 0.679847859 | -0.001093348 |
| 0.5 | 0.677743951 | -0.003197256 |
| 1 | 0.670350354 | -0.010590852 |

All declared nonzero weights reduce the point score. The exact equal-mixture recipe is not promoted. This does not reject differently trained teachers, feature views, residual models or other interactions. The next priority stays the independently trained heterogeneous/history-target student and delivered numerical-embedding/capacity comparisons.

Evaluation uses the original 468 exposed groups, source-qualified joins and official clipping/masks. Component bootstrap is conditional on fixed predictions; it does not estimate training variation or public transfer. No new public best or public forecast follows from this diagnostic.

[Frozen preregistration](H264_FIXED_EQUAL_MIXTURE_PREREG_20261008_R1.json), [all weights and uncertainty](H264_FIXED_EQUAL_MIXTURE_RESULT_20261008_R1.json), [grader](grade_fixed_mixture.py).
