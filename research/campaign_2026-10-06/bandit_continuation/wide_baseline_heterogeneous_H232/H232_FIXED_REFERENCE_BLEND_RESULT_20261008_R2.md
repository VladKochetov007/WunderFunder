# H232 fixed-epoch exposed-fold comparison

Actual own scratch full-feature GRU384x2 fixed epoch3 predictions were collected from Kaggle R8 and independently scored by root against the frozen own GRU256 reference. The four completed epochs used 59,654 updates /26 AMP skips; postfit/export failures were repaired without fitting again.

Official global_wp_clip2_v1,468 excluded sequences /141 components /808,863 scored rows; this is exposed discovery. Raw predictions are blended before official clipping. All declared weights are reported.

| Weight on H232 | Official WP | Delta from reference |
|---:|---:|---:|
| 0.0 | 0.680941206807 | +0.000000000000 |
| 0.1 | 0.680617071449 | -0.000324135358 |
| 0.25 | 0.679752014754 | -0.001189192054 |
| 0.5 | 0.677292303245 | -0.003648903562 |
| 1.0 | 0.668595547728 | -0.012345659079 |

The exact scratch recipe and fixed blends are not promoted. This is not a width-only ablation: reference objective, initialization, feature/teacher history and compute differ. The result does not reject larger recurrent teachers, heterogeneous targets or other interactions. No public forecast is supported.

The first grade stopped before scoring because the subset-only shared alignment was applied to a full-valid export. R2 maps excluded source-qualified keys to actual full-export indices; shared metric, predictions, preregistration and weights are unchanged. An independent actor checked noncontiguous/reordered row mapping and rejection of target-order mismatch. Raw failure and original admission are preserved.

[Complete result](H232_FIXED_REFERENCE_BLEND_RESULT_20261008_R2.json), [root admission](H232_FIXED_REFERENCE_GRADE_ROOT_ADMISSION_20261008_R2.json), [independent alignment review](H232_FULL_EXPORT_ALIGNMENT_PATCH_REVIEW_20261008_R1.md).
