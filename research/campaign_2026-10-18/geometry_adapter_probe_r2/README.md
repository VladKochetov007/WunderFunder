# Frozen-backbone geometry input adapter

**E-GEOMETRY-ADAPTER-PROBE-20261018-R2**, a bounded child of [the quick paired probe](../quick_geometry_probe_r1/README.md). The first pair scored 0.6804165153 for features versus 0.6804784081 for the zero control on 468 sequences / 141 components. Their parent scored 0.6812265132. The feature-control difference is −0.00006189, with a fixed-fit component 95% interval [−0.00012172, −0.00000319]. These conditional results narrow the tested short continuation; they do not cover fitting-seed or transfer uncertainty.

The next question is whether useful added inputs can be learned while preserving the strong backbone exactly. Reinitialize the same parent and zero-extend its input weights. Freeze every parameter except `inp.weight`, mask gradients on its 211 old columns, use fresh AdamW with **weight decay 0**, and train only the 60 new columns. Masking gradients without disabling AdamW decay would still change old weights; this implementation prevents it. [feature_adapter.py](../../../src/wnn/training/feature_adapter.py) runs at model setup and is enabled only by `--research-feature-adapter-only`. No resume path is claimed.

- [recipe.json](recipe.json) contains the exact one-epoch, 4,096-draw FP32 command, 20-minute fit bound and user deadline 23:50 UTC / 02:50 Kyiv.
- [profile_command.json](profile_command.json) specifies the actual-shape five-batch profile, excluding continuation-state saves. `profile.log` preserves finite-update, gradient, timing and memory evidence as it lands; fit admission requires successful completion.
- [cache_handoff_r2.json](cache_handoff_r2.json) records one bounded 3 GB file-cache reclaim after the first pair successfully completed fit/export/grade: parent charge 14.53 → 11.44 GB, no process killed and no limit changed. It reuses the existing memory-snapshot helper, with swappiness 0, to admit the next task conservatively.
- The initializer is the same clean-F0 `warm_mechanism_f0_s3_c_r3@ep1`; preprocessing, auxiliary losses and fitting membership are inherited. Evaluation is the same exposed F0, one seed, fixed epoch 0. This is a separately identified recipe, not a hidden rerun.

No quality result exists yet. Admission follows the profile. After training, verify bitwise equality of every old parent parameter and normalizer, nonzero finite added weights, the actual official metric and native parity before packaging. The direction can remain worth a server study even if this bounded adapter is uncertain or negative.

## Execution and verification files

- [run_adapter.py](run_adapter.py) owns the serial fit, fixed-epoch GPU prediction export and official grade; [progress.jsonl](progress.jsonl) records actual phase timestamps. The controller admitted a 4.5 GB reservation from a 3.186 GB actual-shape profile, one CPU core and no swap. [runtime/README.md](runtime/README.md) indexes resource receipts.
- [check_frozen_parent.py](check_frozen_parent.py) checks every state tensor, the original 211 input columns and normalization bitwise, while requiring finite nonzero appended weights. It exports the checkpoint's actual fitted feature prior for serving. It is run after the GPU worker drains.
- [compare_adapter.py](compare_adapter.py) reuses the first pair's source-qualified alignment and official component-bootstrap tools. The fixed comparison is adapter versus the original parent, on the same 468 exposed-F0 sequences.

Independent source review found no preservation blocker for this exact fresh-AdamW/zero-decay command. Two API issues are outside this frozen recipe: a nonempty generic `--freeze` would be overridden by adapter setup, and the suffix boundary is inferred from extension width rather than validated against explicit feature names. No such generic freeze or altered layout is used here. Add guards before making the option generally reusable; do not mutate running sources. Source review cannot establish native parity or artifact contents.

## Completed result and decision

The fixed-epoch run completed fit, GPU prediction export and official scoring at 22:21:45 UTC on 17 October, then its admission barrier was drained. Training took 693 seconds with 5,120 completed finite updates, zero nonfinite/AMP-skipped updates and no OOM/max events. Peak worker RAM was 4,052,291,584 bytes.

[comparison.json](comparison.json) records adapter WP **0.6804499040** versus parent **0.6812265132**, difference **−0.0007766092** with conditional component-bootstrap 95% interval **[−0.0013528215, −0.0002791334]**. This is a negative result for this short frozen-adapter recipe on the exposed F0, not a rejection of from-scratch features, a different objective, teacher/student interactions or the full geometry family. Component uncertainty excludes seed, adaptive selection and public transfer.

[frozen_parent_check.json](frozen_parent_check.json) proves every old state tensor and original 211 input columns stayed bitwise equal to parent checkpoint `78357eca…`. All 15,360 new weights are finite/nonzero, maximum magnitude 0.0171941. Child checkpoint SHA-256 is `f03785c3dbc364d3d12a0ad0cbe1969a50a95bdc3a95d1f4bd0f4758f654df19`. [geometry_prior.json](geometry_prior.json) is the actual checkpoint's fitting-only prior, exported for native serving; it is not re-fit on evaluation inputs.

The user requested an exploratory quota ZIP by 02:50 Kyiv. Packaging this candidate tests transfer/serving; it is **not** recommended as a demonstrated improvement over the parent or the team's 0.66832 public result. No further local fits are launched. The [B300 draft PR #15](https://github.com/VladKochetov007/WunderFunder/pull/15) owns the broader study.

After the frozen worker completed, reusable adapter setup was hardened to reject generic `--freeze` and layouts other than the audited 211 raw/event + 60 geometry suffix. These post-run guards do not change the fitted result or its frozen source hashes.

[adapter_result.png](adapter_result.png), reproduced by [plot_result.py](plot_result.py), plots the actual parent/adapter scores and conditional component interval. It visually separates the negative tested recipe from unmeasured seed and transfer uncertainty.

## Exact q8 serving replay

[geometry_adapter_r2_q8vnni_build_receipt.json](geometry_adapter_r2_q8vnni_build_receipt.json) binds the 5,323,202-byte ZIP SHA-256 `f4bbd585d933f4494167c60f0e090877128062581842312a43189a566eec646c`, single coefficient-1 member, checkpoint, exported weights, actual prior, callback and native libraries. [geometry_adapter_r2_q8vnni_f0_serving.json](geometry_adapter_r2_q8vnni_f0_serving.json) replays the **exact ZIP** over all 468 source-qualified F0 groups / 9.36 million callback rows / 808,863 scored rows. All required outputs were finite, and the independently rescored producer reference matched FP32 WP 0.6804499040. Served q8 WP was **0.6803847537**, a −0.0000651504 precision/export difference.

Replay took 305 seconds wall, with 813 MB maximum RSS. Callback-only wall cost 27.01 microseconds per row projects to 17.74 minutes over 39.4 million rows on this local CPU at one full core. This is a row-cost estimate, **not** complete competition runtime or a target-host guarantee. Actual-trained FP32 native parity and q16 precision comparison remain separate checks. Prediction payload and source-qualified sidecar are in `cache/pred_geometry_adapter_f0_s3_r2_q8vnni_valid.{npy,json}`; the receipt contains their hashes.

[geometry_adapter_r2_q8vnni_python311_smoke.json](geometry_adapter_r2_q8vnni_python311_smoke.json) records an exact-ZIP offline import/ABI smoke in the cached Python 3.11.16 image, under 0.2 CPU and 512 MiB including swap, with networking disabled and no Torch import. The native constructor and q8 callback returned finite outputs; 99 warmup rows were suppressed per sequence, and synthetic A-B-A replay was bitwise deterministic after sequence resets. This synthetic smoke does not replace actual-data F0 parity or competition-runtime measurement.

## q16x evaluation-only ZIP smoke

[geometry_adapter_r2_q16x_evalonly_build_receipt.json](geometry_adapter_r2_q16x_evalonly_build_receipt.json) binds the evaluation-only ZIP at [/home/vlad/development/wunder/research_tmp/geometry-serving-20261018/geometry_adapter_r2_q16x_evalonly_20261018.zip](/home/vlad/development/wunder/research_tmp/geometry-serving-20261018/geometry_adapter_r2_q16x_evalonly_20261018.zip), SHA-256 `955bff715871ff8a977d96a43f4edb89b1f5ecec72ae8ee49c0c2b53a3b6688b`. [geometry_adapter_r2_q16x_python311_smoke.json](geometry_adapter_r2_q16x_python311_smoke.json) records the same offline Python 3.11.16 ABI/import, finite callback and synthetic A-B-A reset smoke for that exact archive. It passes packaging-level checks only; the actor's actual F0 q16x replay is separate. Any later copy to `submissions/` must retain this ZIP hash.

## Ledger formatting correction

[ledger_schema_correction.json](ledger_schema_correction.json) preserves exact original bytes/line hashes for four newly authored, uncommitted ledger rows that omitted canonical `id` fields, plus full before/after ledger hashes. Those new rows were corrected to restore strict navigation parsing; every pre-existing row remains byte-for-byte unchanged, and no frozen recipe/result was changed. This explicit correction receipt preserves the original malformed records rather than silently discarding them.

## Selected q16 endpoint and actual precision comparison

[geometry_adapter_r2_q16x_f0_serving.json](geometry_adapter_r2_q16x_f0_serving.json) records the exact q16 ZIP over the same complete source-qualified F0: WP **0.6804519697**, versus producer FP32 **0.6804499040** (difference +0.0000020656). All 9.36 million callback rows and 808,863 scored rows passed finite/output checks. Full replay took 633 seconds wall / 628 seconds CPU, with 813 MB maximum RSS. Callback-only wall cost was 61.57 microseconds per row, projecting **40.43 minutes** over 39.4 million rows on this CPU; the CPU-based callback projection is about 43.15 minutes and the full replay-cost extrapolation about 44.4 minutes. These are measured-row-cost extrapolations, not complete competition runtime or target-host guarantees.

Q16 is selected to preserve the fitted FP32 score while retaining measured runtime margin under 60 minutes. The +0.000002 local difference is quantization/numerical noise, not a scientific improvement claim. The exact recipe remains below its original parent. Q8 is retained as a faster historical alternative, not the delivered recommendation.

The selected file is [geometry_adapter_r2_q16x_20261018.zip](../../../submissions/geometry_adapter_r2_q16x_20261018.zip), **5,323,200 bytes**, SHA-256 `955bff715871ff8a977d96a43f4edb89b1f5ecec72ae8ee49c0c2b53a3b6688b`. [geometry_adapter_r2_q16x_submission_alias.json](geometry_adapter_r2_q16x_submission_alias.json) proves the submission path hard-links the exact tested evaluation archive, preserving bytes/inode/hash. The original evaluation-only filename is historical provenance, not a different model. [geometry_adapter_r2_q16x_evalonly_build_receipt.json](geometry_adapter_r2_q16x_evalonly_build_receipt.json) binds its model, checkpoint, prior, callback and native-library hashes. The independently produced [Python 3.11 smoke receipt](geometry_adapter_r2_q16x_python311_smoke.json) passes offline import/ABI, warmup and synthetic A–B–A determinism for these exact bytes. Actual trained FP32 numerical and strict native-cache replay checks remain the final separate receipts at this writing.

## Release and remaining limitations

[RELEASE.json](RELEASE.json) binds the delivered ZIP and source/evaluation/compatibility receipts. Independent review found no concrete blocker to releasing this explicitly requested **exploratory native q16 endpoint**. It is not a new best or a public-score forecast. Nothing was uploaded to the competition portal.

The [predeclared FP32 numerical protocol](geometry_adapter_r2_f32_native_torch_panel_protocol.json) and [actual FP32 native/Torch panel](geometry_adapter_r2_f32_native_torch_panel.json) must be read together: the callback replay completed, but elementwise tolerance `rtol=1e-5, atol=1e-6` **failed**. Maximum absolute difference was 0.0007533, mean 0.0000555, RMS 0.0000764 on 318,416 requested values. The generic panel's `passed` field covers successful replay/output checks, not numerical equivalence. Native construction uses `Activations::Fast`; no Exact-vs-Fast control isolated the cause. No threshold was changed. The shipped native q16 function was itself scored on all F0 sequences; it is not being represented as Torch-equivalent.

The [q16 native-cache replay](geometry_adapter_r2_q16x_native_cache_parity.json) reproduced its own precision-matched cache on eight source-qualified groups with zero numeric difference. Its historical generic field names mentioning `fp32` refer to the supplied reference argument; in this panel that reference is native **q16**, not Torch. Follow the actor's correction/bit-pattern receipts for exact scope. The separate exact-ZIP Python 3.11 smoke verifies synthetic A–B–A reset determinism.

[Navigation build log](navigation_build.log) records a successful research-tree rebuild. [Navigation check receipt](navigation_check_receipt.json) preserves the nonzero exit and tool stderr (the [stdout log](navigation_check.log) is empty) for a stale generated snapshot while live notes/receipts and monitoring logs continued to change; this is a catalog-freshness limitation, not negative model evidence. The narrative folder indexes link the new outputs directly.

## Final signed-zero and tolerance assessments

- [q16 uint32 parity](geometry_adapter_r2_q16x_native_cache_uint32_parity.json) confirms **zero differences in 318,416 aligned float32 bit patterns**, including signed zero, on eight source-qualified groups. This is exact-ZIP native q16 versus its own native q16 cache.
- [FP32 tolerance assessment](geometry_adapter_r2_f32_native_torch_tolerance_assessment.json) confirms **287,989 of 318,416** requested values exceed the preregistered Torch/native elementwise tolerance. The failed comparison is retained unchanged; it does not establish a cause or invalidate the actual native endpoint's directly measured metric.
- [q16 reference-label correction](geometry_adapter_r2_q16x_native_cache_parity_label_correction.json) preserves the original replay receipt and explicitly corrects its generic `fp32_reference` labels to mean the actual supplied **native q16** cache. No score, prediction, ZIP or original receipt was rewritten.

All final release scopes and receipt hashes are collected in [RELEASE.json](RELEASE.json). Local GPU admission is closed and all fits have ended. The ZIP was delivered before the user's 02:50 Kyiv deadline; the portal upload remains with the user.

## User-reported public confirmation

[PUBLIC_RESULT.json](PUBLIC_RESULT.json) records portal submission `14s6l2z7`: public WP **0.66428**, **30m47s**, **626 MB**, reported by the user after submitting the frozen SHA955bff… ZIP. It passes runtime/compatibility but trails the team best0.66832 by0.00404. This is external transfer evidence for the exact R2 endpoint, consistent with its negative local comparison; no broad family rejection or public forecast for other variants follows.
