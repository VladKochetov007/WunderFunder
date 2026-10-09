# E527 independent method audit V1

**Observed:** 2026-10-07 19:00 UTC  
**Scope:** static review of the E527 selection brief, options and priority matrix against the project contract, H230 source/provenance receipts and owned model/data code. No fit, benchmark, provider request, rental, or score run was performed.

## Findings that change selection

### 1. The teacher/student disjointness condition is too strict

The E527 selection brief and C10 test say to use E521 as a student teacher only when its ancestry is disjoint from *both* the student's fitting groups and scored evaluation groups, unless cross-fit targets are available. That would rule out H230's reviewed same-fold design and confuse in-sample distillation targets with held-out evaluation leakage.

H230 explicitly declares the independently trained GRU384 seed pair as its `samefold_baseline` family. Its source manifest describes the baseline as trained on the exact fold-0 fitting groups and permits an allowed fitting subset plus recursively audited initializer/teacher/propensity ancestors, while excluding evaluation-linked and corrupt groups. The reviewed V2 execution contract says fold-0 evaluation groups stay out of fit/cache rows. The actual source review passed with conditions after checking the source-seal map, teacher inspection and ancestor-closure checks. The verified student cache contains 10,535 train-fit and 1,405 valid-fit groups; the source review identifies the separate 468 exposed evaluation groups. This is a declared, reviewed overlap between teacher and student fitting provenance, not evidence that evaluation groups entered the fit cache.

**Smallest correction:** require the complete teacher/source/propensity ancestor closure to exclude the scored evaluation groups for the fixed transfer claim, and require student target-cache rows to match the declared allowed fitting split. Permit teacher and student to share allowed fitting groups. Require cross-fitting only where the experiment needs out-of-fold teacher targets for a particular group. Keep the overlap visible in the recipe because it can affect supervision quality even when it does not invalidate the held-out comparison.

### 2. Raw-input padding does not change the GRU's input width

E527's C4 proposal attributes a cuDNN recurrent alignment miss to H230's 211-wide raw input and proposes zero-padding it to 212 or 216. In the owned implementation, `RNNModel.inp` maps the input to `hidden=256` before calling `self.gru`; the GRU is constructed with `input_size=hidden=256` and `hidden_size=256`. For H230, `train_rnn.py` derives 211 input columns from 112 supplied features plus 99 causal event features. Therefore cuDNN receives width 256, not 211. Batch 16 and recurrent input/hidden widths 256 already meet the cited small alignment multiples. Padding 211 raw features cannot repair a GRU-input alignment fallback.

The 211 dimension is the reduction width of the preceding `Linear(211,256)` GEMM. A zero-column extension to 212/216 could be a separate input-projection GEMM dispatch experiment, but only after a trace identifies that operator as material and shows the relevant kernel path. It is not a cuDNN-GRU alignment test, and its cost must be compared with the dominant recurrent forward/backward work.

**Smallest correction:** change the C4 hypothesis/test label to “input-projection GEMM shape” if that is the measured bottleneck; otherwise drop padding. First record the actual operator/kernel, tensor layout and timing. Preserve exact projection outputs, carried state, gradients and one optimizer update before timing a full step. Do not infer recurrent dispatch from the raw feature count or the presence of a cuDNN call.

### 3. The decoded-feature cache needs a disk gate before it ranks highly

`SequenceFile.read` emits 20,000 rows × 112 float32 features per sequence: 8,960,000 bytes. A contiguous float32 cache of the 10,535 train-fit groups alone is 94,393,600,000 bytes (94.394 decimal GB / 87.911 GiB). H230's complete fitting pool also has 1,405 valid-fit groups, giving 11,940 groups and 106,982,400,000 bytes (106.982 decimal GB / 99.635 GiB). This is feature payload only, before labels, masks, metadata or filesystem overhead. The scored 468-group evaluation partition must stay out of such a training cache.

The last durable H230 fit-stage disk snapshot recorded 65,297,915,904 bytes free at 17:05:19Z; it is older than this audit and is not a live reading, but it is smaller than even the train-only map. H230 already stores a verified 13.373 GB float32 teacher-target cache. A feature map therefore needs an explicit disk, staging and transfer budget; it is not a small extension to that cache.

**Smallest correction:** keep C17 conditional and below completion of the already prepared H230 path. Admit it only after a matched trace confirms repeated Parquet conversion lies on the critical path and a fresh storage/network check supports the exact fit-only payload. Measure `T_build + T_incremental_stage + T_verify + K*T_mmap` against `K*T_parquet` for the actual number `K` of complete passes/fits, with cold/warm cache behavior reported separately. Do not create the full map from the theoretical loader-wait interval alone.

## Corrected decision order

1. Reconcile H230's live owner, durable attempt ledger and host-side GPU lock before allocating anything. The newest readable H230 handoff is timestamped 17:37:14Z and records no fit receipt or run directory at capture, but explicitly says host systemd/lock ownership was unknown from the managed shell. At 19:00Z this shell still cannot establish whether the host changed afterward; the old snapshot is not proof of present idleness.
2. If H230 is still unlaunched and a fresh host admission passes, continue the already authorized serial seed-3 fit and its separately gated export/grade. The same-shape local H230 profile and source/cache checks already exist; a new remote profile is not a prerequisite to complete this local candidate.
3. Consider one matched remote profile only for a concrete upcoming device decision, after fresh provider quota/network, source staging and actual-device admission. The current E527 market cards are not live offers. Do not let remote profiling delay an admitted score-bearing H230 result.
4. Choose one post-profile systems intervention from the observed critical path. C17 needs the byte/storage and measured break-even gates above. C4 needs the input-projection versus recurrent-kernel distinction. Keep zero workers as the baseline and add one only after same-order and memory checks.
5. Preserve the teacher/value lane. Reconcile and independently grade existing E521 outputs against their matched control before funding duplicate width compute; then evaluate the fixed student/blend on the declared teacher-excluded scored unit. Corrected provenance rules do not imply a positive transfer result. Keep the TPU V4 path at offline-preparation status until real-data objective/gradient parity, transport review and fresh quota gates pass.

## Remaining preparation status

The H230 profile/source/cache receipts establish useful local evidence, not current host liveness or fit completion. The E520 V11 packet is assembled offline under root's accepted schema projection, but its packet transport review and provider push remain pending; the assembly receipt still records that explicit `enable_internet=false` is absent from the built metadata. The E463 TPU V4 candidate is also offline-only: its build manifest records no provider status/quota, transport review, push, TPU execution or fit. Neither handoff was changed or launched in this audit.

The broader E527 prioritization remains sensible where it separates systems profiles from scientific precision/batch/optimizer comparisons, keeps unmeasured priors null, avoids ranking GPUs from peak FLOPs or stale marketplace cards, and puts active-owner reconciliation first. No score or public-transfer claim is added here.

## Reviewed sources and checks

- E527 selection brief, optimization options, priority matrix and source ledger; their original files are preserved unchanged.
- Project `AGENTS.md`, ADR0004 and H230 V2 execution contract.
- H230 samefold teacher manifest, actual-source independent review and verified cache-output receipt.
- `src/wnn/nn.py`, `src/wnn/features.py`, `src/wnn/data.py`, and the H230 feature-width construction in `scripts/train_rnn.py`.
- H230 seed-3 recovery handoff and prior fit-stage disk snapshot; E520 V11 offline assembly receipt; E463 TPU V4 offline build manifest.

This was a read-only source/membership/shape audit. The feature-cache byte counts were calculated from `20,000 × 112 × 4 × group_count`; no large cache, model execution, or provider state was touched. Machine-readable evidence and reviewed-file hashes are in [E527_INDEPENDENT_METHOD_AUDIT_V1.json](E527_INDEPENDENT_METHOD_AUDIT_V1.json).
