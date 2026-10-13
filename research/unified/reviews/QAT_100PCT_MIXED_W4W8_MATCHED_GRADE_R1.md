# Full-cohort mixed-W4/W8 QAT versus matched float-PTQ grade — 2026-10-13

**Verdict:** the independent full-fold component grade and bounded fit/export audit pass. The mixed-W4/W8 QAT endpoint scores `0.681420269` WP versus `0.680660270` for its matched FP32-fit/PTQ-mixed endpoint, a candidate-minus-control difference of `+0.000759999`. The paired 141-component bootstrap interval is `[-0.000013193, +0.001655967]`. This is a positive but uncertain single-seed point estimate on exposed F0; it does not establish a reliable gain or promote the endpoint.

## Frozen deployment comparison

The reference is `qat_deploy_f0_s3_100pct_fp32_r1` exported as `input_w4_recurrent_w8`; the candidate is `qat_deploy_f0_s3_100pct_mixed_w4w8_r1`. The frozen grader validated the fixed R2 native-Q8 anchor and each endpoint's own R3 manifest. Both result files report `global_wp_clip2_v1`, exact full-sequence replay, the same ordered 468-group F0 panel, 141 source-qualified components, and 808,863 selected rows. It recomputed each stored score from component sums, checked per-component target/mask sufficient statistics against the anchor, and performed 10,000 paired resamples with seed `20261013`.

| Metric | FP32-fit/PTQ-mixed | Mixed-W4/W8 QAT | QAT − PTQ | Paired component-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| t0 | 0.674184445 | 0.675029550 | +0.000845104 | [−0.000074345, +0.002075067] |
| t1 | 0.687136095 | 0.687810989 | +0.000674894 | [−0.000122713, +0.001476085] |
| WP | 0.680660270 | 0.681420269 | +0.000759999 | [−0.000013193, +0.001655967] |

The [machine grade](receipts/QAT_100PCT_MIXED_W4W8_PAIRED_GRADE_R1.json) binds result/component-sum hashes, the separate reference/candidate manifest hashes (`4d0e8fe7…a09bc24b` and `b19ec408…e027f3a1`), official metric source hash, fold membership, and bootstrap settings. The [fit/export audit](receipts/QAT_100PCT_MIXED_W4W8_MATCHED_GRADE_AUDIT_R1.json) records the actual checkpoint and artifact hashes. The [candidate native result](../../../artifacts/qat_renewable_r1/100pct_mixed_w4w8_deployment_r3/input_w4_recurrent_w8/result.json), [candidate manifest](../../../artifacts/qat_renewable_r1/100pct_mixed_w4w8_deployment_r3/manifest.json), [matched float result](../../../artifacts/qat_renewable_r1/100pct_fp32_deployment_r3/input_w4_recurrent_w8/result.json), [matched float manifest](../../../artifacts/qat_renewable_r1/100pct_fp32_deployment_r3/manifest.json), [fit completion receipt](../../campaign_2026-10-13/qat_renewable_r1/FULL_MIXED_FIT_COMPLETION_R1.json), and [native export receipt](../../campaign_2026-10-13/qat_renewable_r1/NATIVE_100PCT_MIXED_QAT_R1.json) are the underlying records.

## Fit and exact representation audit

Both actual provenance files ([FP32 control](../../../artifacts/runs/qat_deploy_f0_s3_100pct_fp32_r1/provenance.json), [mixed-QAT candidate](../../../artifacts/runs/qat_deploy_f0_s3_100pct_mixed_w4w8_r1/provenance.json)) use seed 3, the same CP5 initializer (`e30577ff…b776eda`), F0 split (`09263ad4…16a3d`), full-cohort subset manifest and member identity (`c8418fce…be1ab377`, `e8cdbba0…f25184a`), propensity checkpoint (`b243b9d4…b517a7`), and strong-teacher cache metadata (`1e4c5c49…e3dc8`). The arguments match except for the run name and QAT treatment (`none` versus `input_w4_recurrent_w8`). Sample-order RNG and normalization hashes also match. Each arm recorded 14,920 optimizer attempts and completed updates, 746 completed batches, zero AMP skips and zero EMA updates. Candidate checkpoint bytes match both the training completion receipt and its deployment manifest.

The candidate manifest labels the export as fitted QAT with exact checkpoint codes/scales and pins their hashes. The frozen NumPy-only packed-export reader decoded `input_w4_recurrent_w8.npz`; all 12 serialized code/scale payload hashes match those manifest hashes, and all 16 reconstructed effective tensors match the checkpoint tensor fingerprints in the manifest. The source snapshot and all four export archive hashes verify. No Torch checkpoint tensors or prediction arrays were loaded. The native mixed archive is 882,802 bytes; the matched float-parent/PTQ archive is 883,002 bytes, so this endpoint-level archive pair differs by 200 bytes.

| Fit scale | Updates | Matched FP32-fit/PTQ WP | Mixed-QAT WP | QAT − PTQ | Paired component-bootstrap 95% interval |
|---|---:|---:|---:|---:|---:|
| [10%](QAT_10PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) | 1,440 | 0.681182863 | 0.681321905 | +0.000139042 | [−0.000481379, +0.000780547] |
| [30%](QAT_30PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) | 4,380 | 0.680562924 | 0.680834028 | +0.000271105 | [−0.000289693, +0.000854484] |
| 100% | 14,920 | 0.680660270 | 0.681420269 | +0.000759999 | [−0.000013193, +0.001655967] |

All three registered mixed-W4/W8 pairs have positive point deltas, with component-bootstrap intervals that cross zero. Fitting membership and update budget change across the 10%, 30%, and full-cohort comparisons, so these rows describe evidence across scales but do not estimate a causal scale response. They are not cross-seed confirmation.

## Scope and limits

F0 has prior research exposure and the full cohort includes the previously used fitting ancestry. The bootstrap conditions on fixed endpoint predictions and resamples evaluation components; it does not estimate fitting-seed variation. It applies `wp_from_sums` to six legacy raw sums, because the archived component arrays do not contain the centered moments used by `GlobalAccumulator.result()`, so interval limits are approximate. Per-component target/mask sufficient statistics do not provide row-level mask hashes. No clean-CV, public-transfer, final-package ranking, or strongest-submission claim follows.

Related experiment: `E-QAT-DEPLOYMENT-20261013-R1`. See the [paired bootstrap method](QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md), [full-cohort uniform-W4 grade](QAT_100PCT_UNIFORM_W4_MATCHED_GRADE_R1.md), and the [QAT campaign index](../../campaign_2026-10-13/qat_renewable_r1/README.md).
