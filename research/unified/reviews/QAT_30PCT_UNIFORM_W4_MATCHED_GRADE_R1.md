# 30% uniform-W4 QAT versus matched float-PTQ deployment grade — 2026-10-13

**Verdict:** the independent paired grade and matched recipe/checkpoint audit pass. Uniform-W4 QAT is `0.0002399` WP above its matched float-PTQ W4 endpoint on this fixed panel. The paired component-bootstrap interval crosses zero. This is a one-seed scale-ladder comparison on exposed F0 after the 30% adaptation fit; it is not clean CV, a public forecast, or a stop rule.

## Endpoints and score

The candidate is the full-fold `recurrent_w4` endpoint for `qat_deploy_f0_s3_30pct_uniform_w4_r1`; the reference is `recurrent_w4` for `qat_deploy_f0_s3_30pct_fp32_r1`, quantized after its FP32 fit. Each endpoint covers the ordered 468-group F0 fold, all 141 components and 808,863 selected rows with exact full-sequence replay under `global_wp_clip2_v1`. The [paired grade receipt](receipts/QAT_30PCT_UNIFORM_W4_PAIRED_GRADE_R1.json) pins each endpoint's own manifest, result and component-sum hashes. The independent consumer validated both manifests/source-file sets, the common ordered groups/components, and exact component target/mask sufficient statistics against the frozen R2 native-Q8 anchor.

The candidate scores `t0 = 0.6741151`, `t1 = 0.6876851`, `wp = 0.6809001`; the reference scores `0.6739023`, `0.6874181`, and `0.6806602`. Candidate-minus-reference deltas are `t0 = +0.0002128`, `t1 = +0.0002670`, and `wp = +0.0002399`. With 10,000 paired component-bootstrap resamples (seed 20261013), the raw-six-sum 95% intervals are `[-0.0004758, +0.0010720]` for t0, `[-0.0003163, +0.0009132]` for t1, and `[-0.0003046, +0.0008922]` for WP.

## Fit and checkpoint match

The actual seed-3 provenance records the FP32 control and uniform-W4 trainer arguments as identical except for run name and `qat`. Both use the same CP5 initializer (`e30577ff…b776eda`), propensity checkpoint (`b243b9d4…b517a7`), split (`09263ad4…16a3d`), 30% subset manifest (`d9d83184…009b33`) selecting 945 components with member SHA (`6a0ff323…5e55ac`), and distillation-cache metadata (`1e4c5c49…e3dc8`). The recorded model-epoch-0 checkpoint hashes match current files. The cache identity points to the same three clean-T3 strong-family teachers at epoch 2 with coefficient 0.1 each.

Both fits selected fixed epoch 0 after 4,380 optimizer attempts/completed updates, 219 completed batches, and zero AMP-skipped updates. Their sample-order records match exactly: 3,517 sampled sequences, batch size 16, no replacement, zero workers, and RNG-state SHA `023e9eb3…b4bc9`. Input-normalization mean/std hashes match (`bb85c72b…a2da8`, `b7306008…e4454c5`). The actual initializer, propensity checkpoint, split, subset manifest, and cache metadata files hash-match the provenance. This is a matched within-30% pair; it does not make the F0 panel pristine.

The candidate manifest SHA is `2ac01b36…68657267`; it binds `model_ep0.pt` SHA `bb19df00…d7dee275`. I independently decoded the native `recurrent_w4.npz` using the frozen R3 NumPy reader, without importing Torch: all 12 encoded code/scale hashes match the manifest, and all 16 effective tensors match `parent.npz` exactly. The archive SHA is `e00a9d26…98b380c`. The reference manifest SHA is `ebfbe18c…70854e2c`; its FP32 checkpoint SHA is `7922c619…c7320cf`.

## Limits

The bootstrap conditions on fixed predictions and resamples the 141 source-qualified components. It applies `wp_from_sums` to six legacy raw sums; centered moments used by `GlobalAccumulator.result()` are absent, so the intervals approximate the exact centered-moment bootstrap. F0 has prior research exposure and the adaptation subset has inherited CP5/teacher/cache exposure. Only one fitting seed is represented. A raw absolute-score comparison against a 10% fit changes both subset coverage and optimizer update count (1,440 versus 4,380); it cannot isolate a scale effect. This grade supports only the matched 30% QAT-versus-PTQ endpoint comparison, not public transfer or general QAT quality.

Related experiment: `E-QAT-DEPLOYMENT-20261013-R1`. See the [paired bootstrap method](QAT_DEPLOYMENT_COMPONENT_BOOTSTRAP_METHOD.md) and [30% mixed-W4/W8 review](QAT_30PCT_MIXED_W4W8_MATCHED_GRADE_R1.md).
