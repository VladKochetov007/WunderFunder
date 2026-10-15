# Frozen-parent anchor recipe R1

[Root full-fit delivery authority](S_PARENT_ANCHOR_FIT_AUTHORITY_R1.json) now releases one fixed two-epoch S fit after the primary G/F/W fits and exports, conditional on fresh shared admission. Its actual resource profile completed50/50 finite updates without AMP skips/OOM. Exact worker vector,6.5GB host/4.3GB allocator/5400s limits, CP1/state validation, full-valid FP32 export and independent grading are owned continuation work. Historical recipe-ready/no-fit wording below describes the preparation stage; no S fit has run yet.

This is a source-bound, no-new-trainer-source recipe for arm S under
`H-FROZEN-PARENT-ANCHOR-20261015`. The direct-parent S child has completed its
bounded 50-update resource profile after C's CP1 export; it has not been fit,
exported, or quality-graded. The [profile contract](S_PARENT_ANCHOR_PROFILE_R1.json)
and [terminal result](S_PROFILE_TERMINAL_RESULT_R1.json) are linked below. The
predeclared parent/T3 endpoint-mixture diagnostic is complete and indexed in
the [campaign README](../README.md); it does not qualify as an S fit result.

The exact two-epoch trainer argument vector and all input/source pins are in
[S_PARENT_ANCHOR_RECIPE_R1.json](S_PARENT_ANCHOR_RECIPE_R1.json). It runs the
immutable trainer from the reviewed GRU-growth worktree at commit
`56d900e3c283512e7128243374cd937c05979359`, with `PYTHONPATH` set to that
worktree's `src` and `WNN_PROJECT_ROOT` set to MAIN. This avoids selecting a
different trainer from MAIN. The vector keeps the R3 C architecture, parent
initializer and preprocessing, F0 fit membership, seed and sampler, AdamW /
OneCycle schedule, FP16 training, two-epoch horizon, fixed epoch-1 checkpoint,
propensity weights, and `.3/.3` auxiliary losses. It replaces the T3 cache
source with the direct existing parent teacher `c20261003_combo_s3_f0*1.0`
and changes only the output run name plus profile-only/unsupported arguments.

The direct-teacher consumer already present in `train_rnn.py` checks each
sidecar's run and split, the split-manifest digest, full canonical row order
against the target cache, and the complete prediction-file hash before fit.
The pinned parent provenance separately shows 10,535 train-fit and 1,405
valid-fit groups, with no overlap between the 1,405 valid-fit groups and the
468 excluded evaluation groups. The full teacher files also contain rows that
are not selected for fitting; the training dataset selects only the clean
fit membership. Their paired FP32-storage arrays total 1,996,640,256 bytes
and are existing read-only mmap inputs, so this recipe creates no new cache.

The parent is not teacher-free: its own four-epoch provenance records earlier
distillation from `c20260929_since_s3_f0` and `c20260929_since_s4_f0` at alpha
`.5`. S initializes from that same epoch-3 checkpoint and uses its stationary
saved predictions as targets. Training uses
`0.1 * clamp(label,-2,2) + 0.9 * clamp(parent_prediction,-2,2)`, with the
existing clipped-label magnitude weights. The matched clean-T3 control uses
`0.8125 * mean(three epoch-2 T3 predictions)` before the same training-time
teacher clipping. Thus S versus C changes the teacher function and target
calibration together. This is an explicit screen of parent-function
anchoring, not an isolated source-only causal contrast.

Two input limitations are carried forward rather than papered over. The
parent train sidecar declares FP16 inference, while its valid sidecar omits
the inference-precision field; exact sidecar and prediction bytes are pinned,
but valid-array precision cannot be recovered from metadata. The parent
prediction exports also lack a feature-only scope marker. The frozen
prediction path consumes features for the output, while the runtime trainer
still enforces sidecar and row-order checks. Do not claim precision parity
with the FP32 T3 teachers or exact parent-forward parity.

Before a fit, root must recheck the source manifest and parent artifacts,
profile or otherwise admit the direct-array hashing/read path, and set the
actual GPU resource envelope. The trainer hashes roughly 2.0 GB of existing
parent predictions at startup. No cache-builder workaround is appropriate:
the generic builder expects four distinct family teachers and the clean-T3
consumer is fixed to three epoch-2 sources and the `.8125` scale. Aliasing a
single parent into either schema would misstate the target. The exact direct
trainer vector is therefore the smallest truthful route.

The recipe fixes an FP32 full-valid prediction export and official
`global_wp_clip2_v1` grading of the same exposed 468-group F0 panel as C,
with the exact parent as an additional reference. Results would represent
one seed and one repeatedly inspected fold, not a fresh holdout or public
forecast. The separately predeclared parent / scaled-T3 fixed mixture has now
been graded against the parent with the existing scorer. It is endpoint-
complementarity evidence only, uses the same exposed panel, and does not block
S; see the grade receipt linked from the campaign README.

## Bounded S profile result

The [prelaunch identity receipt](S_PROFILE_PRELAUNCH_IDENTITY_R1.json),
[independent profile review](../../../unified/reviews/FROZEN_PARENT_S_PROFILE_REVIEW_20261015_R1.md),
and [terminal result](S_PROFILE_TERMINAL_RESULT_R1.json) bind this resource-only
probe to the frozen trainer, data and direct-parent inputs. The trainer
completed 50/50 finite optimizer updates across five B16 full-sequence batches,
with no AMP skips. It took 20.23s, peaked at 5,852,131,328B worker memory,
8,286,826,496B parent memory and 2,178,940,928B observed VRAM; command and
parent OOM deltas were zero. This estimates the direct-parent hashing/read and
training path at profile scale only. No checkpoint, full fit, prediction or
quality result was produced by S.
