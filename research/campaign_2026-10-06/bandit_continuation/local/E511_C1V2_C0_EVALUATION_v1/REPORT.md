# E-511 result: paired minimal C1 continuation versus C0

The fixed-epoch C1 pair scored **0.682176** versus **0.681153** for its same-seed C0 pair on the previously exposed fold-0 evaluation groups, a paired delta of **+0.001023**. The frozen 2,000-replicate connected-component bootstrap gives a 95% interval of **[-0.001473, +0.004619]** and `p(delta <= 0) = 0.3015`. The point estimate is positive, but this comparison does not resolve the effect from zero.

| Contrast | C1 WP | C0 WP | Delta | 95% component-bootstrap interval |
|---|---:|---:|---:|---:|
| Seed 3 | 0.681618 | 0.679952 | +0.001666 | [-0.000663, +0.004193] |
| Seed 4 | 0.681647 | 0.679971 | +0.001676 | [-0.001701, +0.006404] |
| Equal-weight pair | 0.682176 | 0.681153 | +0.001023 | [-0.001473, +0.004619] |

The comparison used 468 sequences, 141 connected components and 808,863 scored rows under `global_wp_clip2_v1`. It conditions on these fixed checkpoints and does not include training-seed or public-transfer uncertainty. Fold 0 has prior exposure, so the result is development evidence rather than fresh confirmation or a public-score forecast.

Both C1 fits and the corrected explicit `model_ep1.pt` prediction exports completed without OOM. The first exports used default `model.pt` and were rejected by the frozen grader's checkpoint-hash gate; they are retained in `../E511_C1_DEFAULT_MODEL_EXPORTS_20261007/`. The corrected exports passed sidecar and row-order checks. The frozen grader spec predates candidate scoring and the grader source hash and report are retained beside this file.

The result supports retaining C1 as a leading lineage for clean-group confirmation, while leaving the numerical public prior null. It neither validates nor rejects weak-member ensembles; E-514 is a separate, prediction-only test.

Primary artifacts: `report.json`, `components.json`, `../C1V2_C0_PAIRED_GRADER_SPEC.json`, `../E511_C1_MINIMAL_PAIRED.json`, and the terminal receipts under `../../runtime/attempts/`.
