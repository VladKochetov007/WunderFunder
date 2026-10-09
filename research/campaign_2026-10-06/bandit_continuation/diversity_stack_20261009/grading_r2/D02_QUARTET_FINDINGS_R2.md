# D02 optimizer/loss quartet findings R2

The frozen 11-endpoint grade completed on the previously exposed fold0 panel. The observed result does not support adding Muon WMSE to the AdamW WMSE control in this one-seed recipe: the preregistered 50/50 blend scores lower than AdamW WMSE, with the paired component-bootstrap interval below zero. This is a conditional fixed-prediction result, not a general rejection of optimizer diversity.

## Scope and checks

- Four independently trained cells with the same seed and matched initialization; two epochs (0 and 1), fixed epoch-1 output checkpoint, and 220 completed updates per arm; 350 training sequences spanning 83 components, with 468 evaluation sequences spanning 141 components. Fit/evaluation components were disjoint.
- Official `global_wp_clip2_v1`; 808,863 selected rows after the fixed score mask. Each arm/blend was averaged from raw FP32 predictions in FP64, cast back to FP32, then evaluated with the official clipping/weighting.
- The actual schedule traces retain distinct raw hashes: AdamW has one optimizer group; Muon has two. Across all 220 updates, clocks and every LR/beta1 group value semantically match the AdamW schedule.
- All 11 official-accumulator scores matched the independent centered formula within 1e-12. Weights were fixed by the frozen preregistration; no posthoc blend search was run.

## All frozen endpoints

| Endpoint | T0 | T1 | WP |
|---|---:|---:|---:|
| AdamW WMSE | 0.591212178 | 0.620791668 | 0.606001923 |
| AdamW Huber δ=1 | 0.583089471 | 0.619388800 | 0.601239135 |
| Muon WMSE | 0.586726131 | 0.614581213 | 0.600653672 |
| Muon Huber δ=1 | 0.581074629 | 0.611037630 | 0.596056129 |
| 50/50 AdamW WMSE + AdamW Huber | 0.588875601 | 0.622084801 | 0.605480201 |
| 50/50 AdamW WMSE + Muon WMSE | 0.589823947 | 0.618460331 | 0.604142139 |
| 50/50 AdamW WMSE + Muon Huber | 0.587678638 | 0.618001453 | 0.602840046 |
| 50/50 AdamW Huber + Muon WMSE | 0.588025163 | 0.619820003 | 0.603922583 |
| 50/50 AdamW Huber + Muon Huber | 0.582976282 | 0.616297932 | 0.599637107 |
| 50/50 Muon WMSE + Muon Huber | 0.585442430 | 0.614356231 | 0.599899330 |
| 25% each: all four arms | 0.587953232 | 0.619057051 | 0.603505142 |

## Primary contrast and uncertainty

Primary: 50/50 AdamW WMSE + Muon WMSE minus AdamW WMSE = -0.001859784 WP (candidate 0.604142139; reference 0.606001923).
Paired whole-component bootstrap: 95% percentile CI [-0.002617604, -0.001124842], bootstrap SD 0.000381081, 2000 replicates, seed 271828; 141 components and 468 sequences. The observed resamples had `p_le_0=1.0000`; this is descriptive and not adjusted across secondary contrasts.

This interval conditions on the four fitted models and this exposed fold. It does not include training-seed variability or public-transfer uncertainty. The score is not a public-score forecast. The negative result applies to this declared recipe and these fixed endpoints; materially different activation, schedule, target, sample or model-family interactions remain untested.

## Exact inputs

- Frozen preregistration: `grading_r1/FROZEN_D02_QUARTET_R1.json` (SHA-256 `b6e8ff174166ae66259b872b6bf3c6809a17fccb3754b472201aaf8fb1f56134`).
- Reviewed quartet bindings: `grading_r2/ROOT_QUARTET_BINDINGS_R2.json` (SHA-256 `af696917ec3bb7447c7da065bb15b4a09fda5ca92a518ce194b25141041fdfe6`).
- Root terminal reviews: `ROOT_GPU_R5_TERMINAL_REVIEW_R1.json` (SHA-256 `60e9770de432a6b39d3ae41c818d60b2d47b5298f630b33ee3483a0e7fecc7f9`) and `ROOT_D02_R5_TERMINAL_REVIEW_R1.json` (SHA-256 `927bd79e3e5bf2bd8ecaecb45c237c63b4a076230ffb51ed2adb8391574e08fe`).
- Actual schedule audit: `grading_r2/D02_SCHEDULE_SEMANTIC_AUDIT_R2.json` (SHA-256 `948c12d4fa4fdced6e70b859310939a075cd2a75970bcd93f93a871758ac401a`).
- Machine-readable grade: `grading_r2/QUARTET_GRADE_R2.json` (SHA-256 `f3c16c6de3df946b89f15270113f8a5a63e806d00a0fefba3afbb4e2eb2909b9`).
- Paired component moments: `grading_r2/QUARTET_GRADE_R2.moments.npz` (SHA-256 `4ad30284f85da89847f5e9231dc8c17978dda46bac8a0ad475f9fe629908a6dc`).
