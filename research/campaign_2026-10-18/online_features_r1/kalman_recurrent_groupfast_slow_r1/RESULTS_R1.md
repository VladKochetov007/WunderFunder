# Recurrent groupwise Kalman-state screen R1

**Status: completed local screen; tiny exposed-F0 point increase with substantial conditional uncertainty. Keep the representation family open; no new B300 draft is justified by this result alone.**

## Question and exact comparison

The earlier groupwise fast/slow Kalman-state weighted-ridge proxy lost against raw inputs, but did not test recurrent nonlinear use. This screen retained all raw and event inputs, appended the 39 causal groupwise states, and compared them with an identical-width all-zero state control. Both arms started from the same `warm_mechanism_f0_s3_c_r3@ep1` checkpoint, used the same 32 complete source-qualified fit components, seed3, batch8, chunk2000, one FP32 epoch, AdamW at `3e-5`, OneCycle, clip norm1, and fixed epoch0 checkpoint. The inherited auxiliary weights were identical: `aux_next=0.3`, `aux_struct=0.3`; the other auxiliary weights were zero.

The subset has 102 train row groups and66 validation development groups from32 components. The outer F0 evaluation groups were excluded from fitting and the initializer. The parent itself was previously trained on the full F0 fit membership, so the paired adaptation tests only the incremental effect of exposing the new states on top of that common parent. It is not a scratch-trained teacher or a seed replication.

## Feature and serving checks

For each input channel, the feature extractor normalizes using only the first99 rows, then computes pre-update diagonal random-walk Kalman innovations. It emits fast and slow group innovation-energy summaries plus their log ratio for the13 input groups. No targets, masks, identifiers or future rows enter the extractor; state resets at each sequence boundary.

On one complete20,000-row train sequence, all 39 states were finite. Batched Torch matched the NumPy implementation within `9.54e-7`; one-row Torch streaming matched the batch path within `9.54e-7`; NumPy streaming matched exactly. The frozen reference extractor differed by at most `1.91e-6`. A perturbation after row10,000 changed the preceding feature prefix by exactly zero, repeated reset output matched exactly, and the99-row warmup output was all zero. The [preflight receipt](preflight_result_r1.json) binds the source sequence and implementation hashes.

The actual B8 CUDA profile completed60 timed loader batches and610 finite optimizer attempts across chunked sequences. Worker peak RAM was4.052 GB under a4.5 GB task cap, peak observed VRAM was1.407 GB, and OOM/max counters were zero. The cgroup recorded memory-high pressure near its configured threshold, so both fit arms were serialized under the same4.5 GB cap. Active and control fits completed in25.2 s each at3.327 GB and2.644 GB RAM peak, respectively; both had zero memory-event deltas. Prediction exports each covered all468 evaluation groups with shape `[468, 20000, 2]`, were finite and feature-only, and had matching evaluation-membership hashes. Prediction/export worker peaks were at most1.581 GB RAM and1.523 GB observed VRAM, with no OOM/max events.

## Official metric results

Scores use the unchanged `global_wp_clip2_v1` metric, its cached official `is_scored` mask, and the fixed F0 `need_prediction AND is_scored` evaluation groups. The two prediction arrays were frozen and hash-checked before the grade script read target or mask arrays.

| Endpoint | WP | Target 0 | Target 1 |
|---|---:|---:|---:|
| Raw + active 39 states | 0.67986413 | 0.67234267 | 0.68738559 |
| Matched raw + zero states | 0.67981250 | 0.67221912 | 0.68740587 |
| Active minus control | **+0.00005163** | +0.00012355 | −0.00002028 |

The paired component bootstrap resampled141 connected evaluation components over2,000 draws. Its conditional 95% interval for the WP difference is `[−0.00013143, +0.00027018]` (`p(delta<=0)=0.3125`). This describes fixed endpoints on repeatedly inspected F0; it excludes fitting-seed, fold, sample-selection and public-transfer uncertainty. It is not a significance or clean-validation claim.

By scored position, active-minus-control WP deltas were:

| Rows | Delta WP | Scored rows |
|---|---:|---:|
| 99–499 | +0.00022656 | 17,198 |
| 500–1,999 | −0.00023541 | 62,742 |
| 2,000–9,999 | +0.00012554 | 316,788 |
| 10,000–19,999 | +0.00004391 | 412,135 |

The pattern is mixed rather than a stable later-sequence improvement. The class deltas are also small and uneven: approximately +0.0000225, −0.0000233, −0.0000184, and +0.0002171 for classes0–3. Training traces show the input-projection gradient squared-share rising from2.97% in the zero control to15.71% with active features, while the readout remains dominant. This confirms the new input block receives gradient, not that it improves generalization. All210 chunk updates in each arm had finite gradients, but clip norm1 was active for100% of recorded steps.

## Interpretation and next allocation

The observed delta is positive but tiny relative to its component-level uncertainty. The four position bins change sign, and only one fitting seed, one exposed fold, and a 32-component incremental fine-tune were measured. This is not a reason to reject the nonlinear/recurrent direction, nor evidence for a B300 gain. The feature scale was not refit on the 32-component subset: newly appended columns inherited zero mean/unit scale, while the fast/slow log-ratio can reach magnitudes near10. Scaling or bounding the appended block is a concrete next local variant to screen, with fit-only statistics and an equal-width zero control.

No new B300 draft was created. Existing [draft #14](https://github.com/VladKochetov007/WunderFunder/pull/14) and [draft #17](https://github.com/VladKochetov007/WunderFunder/pull/17) already cover the broader causal-innovation and Kalman teacher/student research paths. If a bounded fit-only scaling variant or independent fit cell shows a more coherent conditional signal, add the corresponding scope to the existing handoff or make a distinct B300 draft only if the mechanism is genuinely new. The predeclared large-server standard remains30 paired seed/fold cells, full precision, and returned training traces, checkpoints, predictions and grades.

## Provenance and limits

- Frozen recipe: [recipe_r1.json](recipe_r1.json), SHA-256 `5f72067709ca5859dc5d963861771bfc16c92cc1eeb1331d8fa94f618e2126b6`.
- Profile evidence: [profile_receipt_r1.json](profile_receipt_r1.json), SHA-256 `f80abd7c0e778b2b3c280fcb0b721ea53db8eaba87f8f01f67e3beea6cfaea19`.
- Official grade: [grade_result_r1.json](grade_result_r1.json), SHA-256 `e0ca4705c40e6990ebda5d0e09d82745d032b2b7c39984ef88dd81e8d42c6c45`.
- Initializer checkpoint SHA-256: `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`.
- Active/control checkpoint SHA-256: `f75c9ec9e1e0ef5fc42ac95743625c730768028d328741ee002ebf6048850d77` / `53669b65d6cd35bf09cfede6ef91daa3132f827148740a222084166750d29bb1`.
- Active/control prediction SHA-256: `8c4fec371df0443464460b3d94f71d47c482d5641ea1ec3ae395dd0aa02ef3c9` / `6850cbb2d972dee2b692f39b6292fc50c42ad3b2a553fff00055c0231017d2f0`.
- Local controller attempts: `kalman-group-recurrent-profile-20261018-r2`, `kalman-group-recurrent-active-20261018-r1`, `kalman-group-recurrent-zero-20261018-r1`, and the two `kalman-group-recurrent-predict-*` attempts under `research/campaign_2026-10-18/online_features_r1/runtime/attempts/`.

F0 was repeatedly inspected and adaptively used. This screen does not estimate a fresh-holdout delta, fit-seed variance, clean-fold transfer, competition leaderboard effect or online adaptation to observed targets. No submission was built and no data, weights or predictions were uploaded.
