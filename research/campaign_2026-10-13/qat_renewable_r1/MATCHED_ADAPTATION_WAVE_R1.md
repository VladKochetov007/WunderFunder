---
kind: result
status: inconclusive
created_utc: 2026-10-13T17:22:51.723690+00:00
updated_utc: 2026-10-13T17:22:51.723690+00:00
hypothesis_ids: [H-QAT-W4-20261013]
experiment_ids: [E-QAT-WEIGHT-10PCT-F0S3-20261013-R1, E-QAT-WEIGHT-30PCT-F0S3-20261013-R1]
evidence_paths: [LOCAL_10PCT_EXECUTION_R1.json, LOCAL_30PCT_EXECUTION_R1.json]
---

# First matched weight-QAT adaptation wave

Six local GPU fits and their four primary native comparisons are complete. Each comparison uses the same parent, teacher targets, subset, optimizer schedule, final epoch checkpoint and full-F0 evaluation. QAT keeps activations and recurrent state FP32 during fitting; native evaluation adds its existing activation quantization.

| Fitting subset | Format | Floating continuation + PTQ | QAT | QAT minus PTQ | Paired 95% component interval |
|---|---|---:|---:|---:|---:|
| 10% | [Uniform W4](../../unified/reviews/QAT_10PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) | 0.681413210 | 0.681298282 | -0.0001149 | [-0.0009160, +0.0006608] |
| 10% | [Mixed W4/W8](../../unified/reviews/QAT_10PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) | 0.681182863 | 0.681321905 | +0.0001390 | [-0.0004814, +0.0007805] |
| 30% | [Uniform W4](../../unified/reviews/QAT_30PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) | 0.680660185 | 0.680900109 | +0.0002399 | [-0.0003046, +0.0008922] |
| 30% | [Mixed W4/W8](../../unified/reviews/QAT_30PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) | 0.680562924 | 0.680834028 | +0.0002711 | [-0.0002897, +0.0008545] |

All grades cover 468 full 20,000-row sequences, 141 components and 808,863 selected rows. The grader independently checked actual ancestry, masks/target statistics, source and checkpoint identities, update counts, sampling and exact archive codes/scales. The official metric remains `global_wp_clip2_v1`. The 10% fits used 1,440 updates; the 30% fits used 4,380. Their identical within-subset sampling preserves matched comparisons, while cross-subset differences combine changed coverage and optimization duration.

The larger subset improves the QAT-minus-PTQ point effect for both formats, but absolute 30% scores are lower. This does not isolate whether coverage or training duration caused the change. Intervals span zero and do not establish either robustness or failure. They resample fixed evaluation components; they do not measure fitting-seed variation. F0 was previously exposed, and CP5/teacher ancestry saw the full fitting set before these adaptation subsets.

Exact Q4 transport saves archive space while leaving resident byte-kernel weights unchanged. See the [five-format baseline](baseline_results_r2/README.md). The same floating checkpoints are also being evaluated with native Q8 through the [additional endpoint sequence](NATIVE_FLOAT_Q8_SEQUENCE_R1.json), so final ranking includes continued training at the original serving precision.

The [full-data control allocation](ALLOCATION_100PCT_FLOAT_R1.json) follows the planned scale ladder and supplies a common reference for full-data candidate arms. It is currently training; no full-data result or final package is available. Learned-scale implementation has passed focused tests and is under independent review before actual-data profiling. Seed/fold confirmation and compressed fixed-blend comparisons remain pending.

Sources: [frozen 10% commands](LOCAL_10PCT_EXECUTION_R1.json), [30% commands](LOCAL_30PCT_EXECUTION_R1.json), [research plan](PLAN.md), and the independent pair reviews linked in the table. No public upload or public-score forecast follows from these results.
