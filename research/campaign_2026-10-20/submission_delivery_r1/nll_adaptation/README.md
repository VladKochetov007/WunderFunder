# Local NLL adaptation

Frozen two-arm recipe against the H256×2 seed-3 F0 parent. The WMSE and NLL arms share initialization, strict-FP32 update schedule, target cache, and one appended zero-initialized logvariance head; only the objective flag differs. Exact commands and all known pins are in [recipe.json](recipe.json).

Historical launch status: both bounded profiles passed; the matched two-fit sequence was running under `wnn-overnight-nll-fit-20261020.service` (invocation `d2a82026557c4f5b915a361b8f8e375a`). Profile uses 2 warmup plus 4 measured full batches per arm (60 chunk updates total) and omits `--save-training-state`; fit uses two full epochs and fixed zero-based checkpoint epoch 1. Root owns launch and monitoring.

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

## Completed control and source refresh — October21 Kyiv

The [control completion receipt](control_terminal_progress_r1.json) records both
fixed epochs:4249.17seconds fit walltime,14920completed optimizer updates, zero
nonfinite gradient steps and zero skipped updates. Epoch1 preclip gradient median
was0.60970,p95was1.07604,clip rate0.0698 and mean clip factor0.9879. The NLL
arm remains live under the same fit service; the frozen seven-endpoint scorer
will run after both fits finish. No predictive improvement is established yet.

A fresh GitHub/remote inspection found research-team835f35ef6 adds dictionary
health evidence, with no terminal grade. PR32 and amended PR28 provide unrun
Transformer/runtime handoffs. They do not add a measured model-selection result.
Root is preparing a conditional F1/seed4 matched pair and production pathway
while this fit completes; these are not admitted jobs and preserve the15-hour
allocation. See the frozen F0 plan and fold catalog before spending follow-up.

## First NLL epoch optimization diagnostics

The [matched epoch0 receipt](nll_epoch0_diagnostics_r1.json) records NLL epoch0
completion in2255.2seconds with7460successful updates, no nonfinite gradients
and no skipped updates. Serialized sample-order and normalization identities
match the control exactly. NLL preclip norm median33.75702,p95was91.06715; all
updates clipped, with mean factor0.0418, versus control0.9866. The head accounts
for95.78%of squared gradient norm on average, but this aggregate includes mean,
auxiliary and variance rows and cannot identify which rows cause the change.
Negative NLL loss is mathematically permitted and is not a quality score.

This diagnoses a substantial objective/clipping interaction. The current
comparison remains frozen through epoch1 and validation; no LR, clipping,
variance constraint or checkpoint rule was changed. If endpoint evidence is
weak, distinguish an optimization limitation of this recipe from a rejection
of heteroscedastic learning before allocating follow-up.

## Both fits completed; corrected export attempt

The [terminal fit receipt](fit_terminal_r1.json) binds both CP1/provenance/history
files. Control took4249.17seconds and NLL4391.30seconds; the fit worker took
8640.57seconds total. Each completed14920updates with zero nonfinite-gradient
steps and zero skipped updates. Serialized sampling and normalization identities
match between arms in both epochs. NLL epoch1 still clipped all updates (median
preclip28.3077,mean factor0.039); this is not a quality result.

The initial export attempt failed before any prediction: its CLI named the root
evaluation-manifest copy, while the frozen source pin required the producer
copy. Their bytes have the same94d22268…SHA. Preserve [failed attempt report](post_fit_plan_report.json),
`exports.log` and original plans. The [R2 plan](post_fit_plan_r2.json) corrects
only those manifest CLI paths and uses distinct audit/report/log paths. The
[amended admission](post_fit_admission_r2.json) pins its [launcher](run_post_fit_r2.sh),
retains original execution pins and accounts for the0.2second failed attempt.
No predictor, metric, endpoint, coefficient or checkpoint selection changed.

Active corrected supervisor: `wnn-overnight-nll-postfit-r2-20261021.service`,
invocation65bdd8af278c41ed8f916741c323ff5a. Export worker:
`wnn-overnight-nll-export-r2-20261021.service`. Its
[actual preflight pass](prediction_preflight_r2.json) binds source/data, all
completed checkpoints/provenance/ancestry, row order and absent outputs. First
WMSE export is running; no score or new candidate is claimed yet. Continue from
`exports_r2.log`, `post_fit_plan_report_r2.json`, `export_resource_samples_r2.log`
and the unchanged `grades/f0_seven_endpoints_r1.json` destination.

The [independent R2 repair review](export_repair_review_r2.json) confirms the
manifest-path/receipt-only change preserves all scientific and resource
settings. Its source-gate caveat is covered by the actual pre/post validator,
which also checks ignored untracked Python sources. Both child prediction
exports completed; old-Z4 export and the final audit/grade remain in progress.

## Completed F0 endpoint evaluation

[Actual results](RESULTS_F0_R1.md) report all seven frozen scores and paired component intervals. Best point endpoint is75%oldZ4/25%WMSE, WP0.6835218031 versus oldZ4 alone0.6831643304; the NLL blend is0.6835179511. NLL minus WMSE is−0.0000436109. Both fits, all three exports, artifact audit and official grading completed. These are one exposed-fold/one-seed results with broad conditional intervals; no new ZIP or TL3 improvement is established. [R2 chain report](post_fit_plan_report_r2.json), [post-export audit](prediction_post_export_r2.json), [start receipt](post_fit_start_r2.json), [export log](exports_r2.log), [resource samples](export_resource_samples_r2.log), and [supervisor log](post_fit_supervisor_r2.log) preserve the successful repaired attempt. Actual endpoint review is pending.

[Grade navigation](grades/README.md) indexes the retained official result payload.

[Independent actual evidence review](ACTUAL_EVIDENCE_REVIEW_R1.json) passes source/metric/ancestry and result-scope checks. The apparent parent discrepancy was a different historical checkpoint, not a discovered mask/cache shift. No new fits or array loads were used for this review.
