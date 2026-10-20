# Local NLL adaptation

Frozen two-arm recipe against the H256×2 seed-3 F0 parent. The WMSE and NLL arms share initialization, strict-FP32 update schedule, target cache, and one appended zero-initialized logvariance head; only the objective flag differs. Exact commands and all known pins are in [recipe.json](recipe.json).

Status: both bounded profiles passed; the matched two-fit sequence is running under `wnn-overnight-nll-fit-20261020.service` (invocation `d2a82026557c4f5b915a361b8f8e375a`). Profile uses 2 warmup plus 4 measured full batches per arm (60 chunk updates total) and omits `--save-training-state`; fit uses two full epochs and fixed zero-based checkpoint epoch 1. Root owns launch and monitoring.

Accounting: 2 new scientific fits total, one per arm; no new teacher fits. The target cache stores 0.8125 times the mean of three fixed MainT3 teacher epoch-2 predictions; keep cache scale 1.0. Parent/cache artifact names and target semantics are pinned in the JSON. A positive small CPU MLP NLL pilot is only a directional prior and does not establish GRU transfer.

This is a strict-FP32 adaptation from a historical FP16 parent, not a fresh-FP32 replication. The one-seed/one-fold comparison is directional and F0 is exposed through ancestry. Root's grader should score parent, each child and the fixed 50/50 parent/NLL prediction blend without coefficient search. Prior parent FP16 time was 932–1086 seconds per epoch; the new strict-FP32 profiles measured about1.68seconds per full batch, projecting42minutes per two-epoch fit. This is a short-profile projection, not a measured fit duration.


## Execution evidence and navigation

- [Original preparation](recipe.prepared.json) and [admitted recipe](recipe.json): same scientific parameters; root reduced allocator cap to4.5GB and bound unchanged Parquets by size/mtime/footer identity. The recipe retains its original preparation status; the dated admission is authoritative.
- [Input verifier](verify_inputs.py) and [actual input verification](input_verification.json): source, checkpoints, target and propensity caches verified before profiles. Four propensity digest placeholders in the recipe are resolved in this receipt.
- [Profile admission](profile_admission.json), [literal profile plan](profile_plan.json), [raw profile log](profiles.log), [sequence report](profile_plan_report.json) and [profile result](profile_result.json):60/60updates per arm, finite losses and preclip gradients, no AMP skips;41.30seconds total,3.49GB peakRSS, zero swap. Final parameter/optimizer-state finiteness and CUDA peak were not emitted by this generic profile path.
- [Fit admission](fit_admission.json) and [literal fit plan](fit_plan.json): one new fit per arm, no new teachers, three-hour maximum for the pair. Practical screen effect0.0002WP is a follow-up guide, not a significance threshold. Latest progress is in [fit log](fits.log) and [sequence report](fit_plan_report.json); existence of a running worker is not a completed result.

The NLL profile clipped every update (mean clip factor0.176), compared with11.7% for WMSE. Nearly98% of squared gradient mass was in the readout, initially learning variance from zero logvariance. This is an observed optimization difference to track, not a demonstrated gain or an automatic rejection. The declared objective comparison includes this interaction with common global clipping; changing variance initialization or per-head clipping would be a separate follow-up.


## Live resource protection

[Resource samples](resource_samples.jsonl) record observed full-fit device use around2.49GB, workerRAM3.2–3.5GB, zero worker swap and the historical cumulative parent cgroup counters (these are not new event deltas). The parent approached4.989GB because cold file cache from preparation remained charged. Root reclaimed666.8MB of clean file cache with `swappiness=0`; parent usage fell from4.662GB to3.999GB without killing a process or changing global swap settings.

The [scoped file-cache guard](filecache_guard.sh), adapted from the existing campaign guard, reclaims only available inactive file cache above4.4GB toward4.0GB. It exits when this fit worker exits and has a3-hour cap,2%CPU,30MBRAM and zero swap. [Its log](filecache_guard.log) records actual reclaim events. This operational change does not alter the training recipe or source. Parent memory remains separately monitored; the worker's hard reservation is4.2GB.


## Pre-outcome endpoint amendment

Before any new validation score, root added the stronger existing old-Z4 endpoint and two fixed mixtures:75%old-Z4+25%WMSE and75%old-Z4+25%NLL. These answer whether the new objective contributes to a stronger deployable system; the original matched NLL/WMSE comparison and50/50parent-NLL remain. There are seven total endpoints and two new scientific fits, with no coefficient search. The independent critic audited13run records, including8teacher sources, and found no source-qualified F0 fitting/evaluation overlap. This does not remove historical panel exposure.

[Evaluation membership](evaluation_manifest.json) binds468valid-source groups/141components. [Prediction and scoring plan](score_plan_r1.json), imported unchanged from actor commit563395207, freezes source hashes, literal commands and the [seven-endpoint screen](score_screen_f0_r1.json). New child exports and a unique-name old-Z4 re-export use strictFP32 inference; old-Z4's historical FP16 cache remains untouched. ParentCP5's existing FP32 cache was byte/order checked. The actual canonical predictor uses B16/C4000 without gradients; analytical memory estimates are in the plan and must be distinguished from measured prediction peaks.

The source worktree now has a documentation-only descendant commit563395207 of68ff6829; no training/prediction implementation byte changed. Training provenance binds module hashes, preserving equal code identity across both arms. Original root fit/profile recipes were retained rather than overwritten by the actor's older prepared copy.

## Throughput update and conditional follow-up

At 20:55 UTC on October20, the control reached epoch0 batch480/746. Sustained
throughput was about5.8 sequences/second, below the short warm-profile estimate.
The revised planning estimate is65–75minutes per two-epoch fit, or roughly
2–2.5hours for the pair before export and grading. This is an estimate, not a
completed duration; the three-hour pair cap and15-hour campaign allocation remain.

The independently inspected [fold asset catalog](FOLD_ASSET_CATALOG_R1.md)
records an available F1 parent, compatible teacher cache and propensity model
for a possible second matched pair. Its parent has23 outputs, so adding two
variance outputs is compatible; the related31-output family student is not a
drop-in initializer. F2 lacks a matching full teacher target bank. The catalog
also identifies an all-data production lineage, which cannot support a held-out
comparison. This is metadata evidence from critic commit87746a30; no follow-up
fit is admitted until the current comparison is graded.

[Control epoch0 progress](control_epoch0_progress.json) records the first completed
epoch:2100.9seconds,7460completed updates, zero nonfinite gradients or skipped
updates, clip rate0.0757 and mean clipping factor0.9866. Median preclip norm was
0.61539 and its95th percentile1.10824. These are optimization diagnostics with
evaluation withheld; the prescribed epoch1 checkpoint remains unfinished.

## Admitted post-fit delivery

The [post-fit admission](post_fit_admission.json) records the durable supervisor
`wnn-overnight-nll-postfit-20261020.service`, invocation
`aaa57a282722430389732e899c97267f`. It waits for this exact fit invocation to
exit successfully and for both fit report entries to be complete. It then
checks the pinned execution files and remaining15-hour allocation before
starting one45-minute maximum export/score worker. The supervisor adds no fit,
adaptive selection or upload. Its [launcher](run_post_fit.sh) and [six-step
plan](post_fit_plan.json) are hash-bound in the admission.

The [audit CLI](validate_prediction_artifacts.py), [input/source checks](prediction_artifact_validation.py)
and [checkpoint/export checks](prediction_export_audit.py) were integrated from
actor stack through `d3055423a`. Preflight verifies the completed checkpoints,
every explicit fitting argument, actual initializer/propensity/teacher-cache
ancestry hashes, source-qualified membership, source tree, Parquet identity and
output absence. Post-export compares the same bindings, then checks FP32
feature-only metadata, full row order, shapes, finiteness and actual byte hashes.
The [operator command reference](EXPORT_SCORE_COMMANDS_R1.md) preserves the
underlying commands; the admitted plan uses the packet's root receipt paths.

Independent Luna/max static review found no remaining concrete blocker after
correcting the initially incompatible `--create-only` proposal and strengthening
the pre-export argument/ancestry/source/mtime checks. The pinned predictor rejects
that flag in full-valid feature-only mode, so output absence plus the shared
single-writer lock protects these unique destinations. This is static review,
not a completed prediction or score. Scoring remains conditional on all gates.

Actual progress will appear in `post_fit_supervisor.log`, `post_fit_start.json`,
`exports.log`, `export_resource_samples.log`, `post_fit_plan_report.json`,
`prediction_preflight_r1.json`, `prediction_post_export_r1.json`, and
`grades/f0_seven_endpoints_r1.json` as the stages execute. Resource samples
measure device usage rather than Torch allocator peak. The worker has4.0GB
hardRAM,1CPU,zero swap and a4.5GB CUDA allocator cap; the aggregate5GB limit
and inactive-file-cache guard remain. Creating `STOP_POST_FIT` before admission
or stopping the waiting supervisor cancels this successor without killing fits.
Root must independently interpret the completed score before another fit or
candidate promotion; the second-fold catalog remains conditional.
