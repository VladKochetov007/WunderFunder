# Final stationary-teacher blend grade R4

The independent R4 grader completed all five fixed blends after adapting its admission reader to the actual R2 native-export receipt. The original R3 schema rejection is retained as a pre-result interface finding. R4 preserves the same member order, equal 0.25 coefficients, float32 accumulation order, F0 group list, and `global_wp_clip2_v1` metric. The frozen [R4 recipe](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/FINAL_BLEND_GRADE_RECIPE_R4.json), grader source, admission compatibility check, and [machine grade](receipts/QAT_FINAL_SELF_DISTILL_BLEND_GRADE_R4.json) are hash-bound.

| Fixed blend | WP | Δ versus baseline |
|---|---:|---:|
| Four-member baseline | 0.7022662377 | — |
| Prior seed-3 learned-W4 QAT replacement | 0.7017641893 | −0.0005020484 |
| Stationary-teacher seed-3 FP32 continuation, native Q8 | 0.7023745300 | +0.0001082923 |
| Stationary-teacher seed-3 learned-W4 QAT | 0.7022161445 | −0.0000500931 |
| Same-checkpoint FP32-to-W4 PTQ | 0.7021328288 | −0.0001334089 |

The learned-W4 blend is +0.0000833157 WP above its same-checkpoint FP32/PTQ-W4 control, but −0.0001583854 below the matched FP32/native-Q8 blend and −0.0000500931 below the fixed baseline. It is +0.0004519552 above the earlier seed-3 QAT blend. Among these five preregistered points, the matched FP32/native-Q8 blend scores highest; this panel does not rank every possible endpoint. The exact learned-W4 blend vector is [blend_r2_qat_w4.npy](../../../artifacts/qat_renewable_r1/final_blend_grade_r4/blend_r2_qat_w4.npy), SHA256 `f8ac48c6fa8f3e12dab4d99e97f3212f3a38a61ddded3ced3977dfc97d6de198`.

All three new member exports completed with exit status 0. The [completion receipt](../../campaign_2026-10-13/qat_renewable_r1/final_self_distill_r2/NATIVE_EXPORT_COMPLETION_R2.json) has SHA256 `24618c0fae22583a17571ed1f7d40234d95fa7f7968b3c780ff3f499623ad7d5`. An independent cross-check matched its result JSON hashes, prediction hashes, weights and source identities, all 468 ordered groups, terminal status, and reset-replay flags against the producer files and R4 grade. Prediction arrays were not reread for that receipt cross-check; the R4 grader had already verified each full prediction hash and independently recomputed each member score against the producer record.

The grade covers 468 ordered F0 sequences and 808,863 selected rows. F0 is exposed: the final parent was initialized from supplied training that includes these validation groups, so these are fitted-data diagnostics, not clean CV, a held-out estimate, or a public forecast. This is one fitting seed and a point comparison; no bootstrap interval or cross-fold confirmation was computed. The native member callbacks passed bitwise reset replay, but their concurrent-export `callback_seconds` values are not controlled runtime comparisons. Root owns final ZIP callback parity, measured serving runtime, and package selection.

Related evidence: [R3 schema rejection](receipts/QAT_FINAL_SELF_DISTILL_BLEND_GRADE_R3_SCHEMA_REJECTION_R2.json), [R4 admission compatibility review](QAT_FINAL_SELF_DISTILL_BLEND_GRADE_R4_COMPATIBILITY.md), [fit ancestry audit](receipts/QAT_FINAL_SELF_DISTILL_FIT_AUDIT_R2.json), [native export completion](../../campaign_2026-10-13/qat_renewable_r1/final_self_distill_r2/NATIVE_EXPORT_COMPLETION_R2.json), and [grade output](../../../artifacts/qat_renewable_r1/final_blend_grade_r4/result.json). `E-QAT-FINAL-SELF-DISTILL-20261013-R4-GRADE`.
