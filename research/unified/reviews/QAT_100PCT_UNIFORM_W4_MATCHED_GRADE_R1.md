# Full-cohort uniform-W4 QAT versus matched float-PTQ deployment grade — 2026-10-13

**Verdict:** the independent full-fold grade and bounded fit/export audit pass. The 100% uniform-W4 QAT endpoint scores `0.681362789` WP versus `0.680741171` for the matched FP32-fit/PTQ-W4 endpoint, a candidate-minus-control difference of `+0.000621618`. The paired component-bootstrap 95% interval is `[-0.000401165, +0.001686432]`, which crosses zero. This is a positive one-seed point estimate on exposed F0, not evidence of a reliable gain or a promotion decision.

## Frozen deployment comparison

The reference is `qat_deploy_f0_s3_100pct_fp32_r1` exported as `recurrent_w4`; the candidate is `qat_deploy_f0_s3_100pct_uniform_w4_r1` exported with its checkpoint's exact recurrent-W4 codes and scales. The full-fold grade receipt validates the frozen R2 native-Q8 anchor and both endpoint-specific manifests. Both endpoint results use `global_wp_clip2_v1`, exact full-sequence replay, the same ordered 468-group F0 panel, 141 source-qualified components, and 808,863 selected rows. The grader checked component target/mask sufficient statistics against the anchor and independently recomputed each stored point score from the component sums.

| Metric | Float-fit/PTQ-W4 | Uniform-W4 QAT | QAT − PTQ | Paired component-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| t0 | 0.674331983 | 0.674999560 | +0.000667577 | [−0.000534745, +0.002007699] |
| t1 | 0.687150359 | 0.687726018 | +0.000575659 | [−0.000564365, +0.001685460] |
| WP | 0.680741171 | 0.681362789 | +0.000621618 | [−0.000401165, +0.001686432] |

The frozen [grade receipt](receipts/QAT_100PCT_UNIFORM_W4_PAIRED_GRADE_R1.json) records result and component-sum hashes, the two endpoint manifest hashes (`4d0e8fe7…a09bc24b` reference, `2627b6a6…e9b4e231b` candidate), metric source hash, and exact bootstrap settings: 10,000 paired resamples of 141 components, NumPy seed `20261013`. The [provenance/export audit](receipts/QAT_100PCT_UNIFORM_W4_MATCHED_GRADE_AUDIT_R1.json) records the independently checked fit and representation identities.

## Fit and exact representation audit

Both runs use seed 3, the same CP5 initializer (`e30577ff…b776eda`), F0 split (`09263ad4…16a3d`), full-cohort manifest and ordered membership (`c8418fce…ab1e377`, `e8cdbba0…f25184a`), propensity checkpoint (`b243b9d4…b517a7`), strong-teacher distillation-cache metadata (`1e4c5c49…e3dc8`), and normalization mean/std hashes (`bb85c72b…c5a2da8`, `b7306008…e4454c5`). The trainer arguments differ only in run name and `qat` (`none` versus `recurrent_w4`); the nested sample-order identity is identical, including the 16-sequence batch, no-replacement sampler, 11,940 sampled sequences, and RNG-state hash `023e9eb3…de4bc9`.

Each run records 14,920 optimizer attempts and completed updates, 746 completed batches, zero AMP-skipped updates, and zero EMA updates; both select fixed epoch 0. The deployment manifests pin the actual `model_ep0.pt` bytes and agree with the corresponding provenance hashes. I read no checkpoint tensors through Torch. Instead, the frozen NumPy packed-export reader decoded the QAT `recurrent_w4.npz`: all 12 code/scale payload hashes matched `checkpoint_quantization.encoded_sha256`, all 16 reconstructed effective tensors matched the manifest's checkpoint tensor fingerprints, and the complete archive SHA-256 matched (`c32fc576…e9151`). Its manifest semantics identify this endpoint as fitted QAT with exact checkpoint codes/scales. The float endpoint's manifest separately identifies W4 as post-training quantization from its effective checkpoint.

## Scope and limits

The paired point difference compares the declared weight-QAT treatment against its matched same-seed float-fit/PTQ control at the same full training cohort and 14,920-update budget. It does not isolate fit-subset scale relative to 10% or 30% runs, which have different update counts. F0 has prior research exposure; this is one fitting seed and the bootstrap conditions on the fixed predictions. The intervals rescore six legacy raw sums with `wp_from_sums`; centered moments used by `GlobalAccumulator.result()` are absent, so the interval is a descriptive approximation. Per-component target/mask sufficient statistics are not row-level mask digests. No clean-CV, cross-seed, public-transfer, package-ranking, or strongest-submission claim follows from this grade.

Related experiment: `E-QAT-DEPLOYMENT-20261013-R1`. See the [paired bootstrap method](QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md) and the [QAT campaign index](../../campaign_2026-10-13/qat_renewable_r1/README.md).
