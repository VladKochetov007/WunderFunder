# 30% mixed-W4/W8 QAT versus matched float-PTQ deployment grade — 2026-10-13

**Verdict:** the independent paired grade and matched recipe/checkpoint audit pass. Mixed-W4/W8 QAT is `0.0002711` WP above its matched float-PTQ mixed endpoint on this fixed panel. The paired component-bootstrap interval crosses zero. This is a one-seed scale-ladder comparison on exposed F0 after the 30% adaptation fit; it is not clean CV, a public forecast, or a stop rule.

## Endpoints and score

The candidate is the full-fold `input_w4_recurrent_w8` endpoint for `qat_deploy_f0_s3_30pct_mixed_w4w8_r1`; the reference is the same deployment case for `qat_deploy_f0_s3_30pct_fp32_r1`, quantized after its FP32 fit. Each endpoint covers all 468 ordered F0 groups, 141 components and 808,863 selected rows, with exact full-sequence replay under `global_wp_clip2_v1`. The [paired grade receipt](receipts/QAT_30PCT_MIXED_W4W8_PAIRED_GRADE_R1.json) pins both independent manifests, result files and component-sum archives. Both passed source-hash verification, ordered fold/component validation, and exact component target/mask sufficient-stat checks against the R2 native-Q8 anchor.

The candidate scores `t0 = 0.6740637`, `t1 = 0.6876044`, `wp = 0.6808340`; the reference scores `0.6737311`, `0.6873947`, and `0.6805629`. Candidate-minus-reference deltas are `t0 = +0.0003325`, `t1 = +0.0002097`, and `wp = +0.0002711`. With 10,000 paired component-bootstrap resamples (seed 20261013), raw-six-sum 95% intervals are `[-0.0002759, +0.0010502]` for t0, `[-0.0004264, +0.0008368]` for t1, and `[-0.0002897, +0.0008545]` for WP.

## Fit and checkpoint match

The actual seed-3 provenance records the FP32 control and mixed-W4/W8 trainer arguments as identical except for run name and `qat`. Both use the same CP5 initializer (`e30577ff…b776eda`), propensity checkpoint (`b243b9d4…b517a7`), split (`09263ad4…16a3d`), 30% subset manifest (`d9d83184…009b33`) selecting 945 components with member SHA (`6a0ff323…5e55ac`), and distillation-cache metadata (`1e4c5c49…e3dc8`). The recorded epoch-0 checkpoint hashes match current files. The cache identity names the same three clean-T3 strong-family teachers at epoch 2 with coefficient 0.1 each.

Both fits selected fixed epoch 0 after 4,380 optimizer attempts/completed updates, 219 completed batches, and zero AMP-skipped updates. Their sample-order identities match exactly: 3,517 sampled sequences, batch size 16, no replacement, zero workers, and RNG-state SHA `023e9eb3…b4bc9`. Normalization mean/std hashes are `bb85c72b…a2da8` and `b7306008…e4454c5`. The actual initializer, propensity checkpoint, split, subset manifest, and cache-metadata files hash-match recorded provenance.

The QAT manifest SHA `09f542ed…1222476` binds `model_ep0.pt` SHA `fce1abad…05280b92`; the provenance SHA also matches the actual file. I independently decoded the candidate's `input_w4_recurrent_w8.npz` with the frozen R3 NumPy reader and no Torch import: all 12 encoded code/scale hashes match the manifest and all 16 effective tensors match `parent.npz`. The archive SHA is `be90f915…f72839e`; the manifest labels this case `exact_checkpoint_codes_scales`. The FP32/PTQ reference manifest SHA is `ebfbe18c…70854e2c`.

The receipt also includes a secondary 30% floating-parent mixed-PTQ versus frozen R2 native-Q8 comparison: WP `-0.0005725`, raw-sum interval `[-0.0019464, +0.0007973]`. This is a different checkpoint and fit ancestry, so it is a context diagnostic, not an isolated precision effect or matched training comparison.

## Limits

The bootstrap conditions on fixed predictions and resamples the 141 source-qualified components. Intervals use `wp_from_sums` over six legacy raw sums; the centered moments used by `GlobalAccumulator.result()` are absent, making these descriptive approximations to the exact centered-moment bootstrap. F0 is exposed and the adaptation subset inherits CP5/teacher/cache exposure; this is one fitting seed. Absolute comparisons with 10% fits also change subset coverage and optimizer count (1,440 versus 4,380), so they cannot isolate a scale effect. Neither the paired grade nor the secondary native-Q8 diagnostic establishes public transfer or general QAT quality.

Related experiment: `E-QAT-DEPLOYMENT-20261013-R1`. See the [paired bootstrap method](QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md) and [30% uniform-W4 review](QAT_30PCT_UNIFORM_W4_MATCHED_GRADE_R1.md).
