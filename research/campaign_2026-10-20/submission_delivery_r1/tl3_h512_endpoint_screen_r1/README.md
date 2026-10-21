# H512 replacement in the complete TL3 endpoint

**Status: NOT_ADMITTED.** This packet prepares a code path and fixed screen; it
does not fit a model, grade a candidate, or admit a submission. Root must first
accept the H512 CP5 terminal receipt and review each package/replay result.

The screen asks whether an H512-trained plain student can replace TL3's
`weights_s5.npz` member while keeping the other two members, their coefficients,
the original native library, and wrapper fixed. It compares the cached original
TL3 endpoint with two exact packages: the admitted H512 CP5 checkpoint and the
predeclared equal mean of CP3/CP4/CP5. No model is fitted here and no blend or
epoch is searched.

## Fixed evaluation

The shared [TL3 F0 replay](../../../campaign_2026-10-18/online_features_r1/tl3_replay/README.md)
supplies feature-only input validation, the native callback, full-sequence
replay, and A-B-A state-reset checking. The new replay script writes its own
endpoint receipts so the legacy receipt's original-TL3 label is never reused.
It verifies the candidate ZIP and build receipt before and after replay, and
requires the package's H512 checkpoint and provenance to match the accepted
CP5/SWA345 source receipt. The `production_candidate` artifact label by itself
does not pass this gate.

The grader checks the frozen 468-group, 141-component F0 panel and 808,863
selected rows. It maps the pinned `valid_targets.npz` using the existing stored
NPZ memory mapper, intersects its `is_scored` mask with the official `WARMUP=99`
prediction requirement, and copies only those selected sequences. Absolute scores use
`wnn.gmetric.global_wp`; paired deltas use
`wnn.evaluation.paired_component_moments` and
`paired_component_bootstrap` with 2,000 component resamples and seed 271828.
The fixed comparisons are:

- H512 CP5 replacement minus cached original TL3
- H512 SWA345 replacement minus cached original TL3
- H512 SWA345 replacement minus H512 CP5 replacement

F0 is exposed, and the two retained TL3 members have unknown training ancestry.
Every complete-ensemble score and delta is therefore a fitted diagnostic, not
clean cross-validation evidence. SWA345 is a dependent derivation from CP5, not
an independent fit.

## Runnable protocol

The exact commands and artifact paths are in
[ENDPOINT_SCREEN_PLAN_R1.json](ENDPOINT_SCREEN_PLAN_R1.json). Run them only after
root accepts the F0 fit terminal receipt and approves the source package path.
The two package builds, native replays, and final grade are create-only. They
write packages, prediction arrays, sidecars, and receipts on physical scratch
under `/home/vlad/development/wunder/research_tmp`.

`replay_h512_tl3_endpoint.py` reuses `f0_source.validate_f0_source` and
`f0_replay.replay_groups`, processes all 99 warmup rows, and requires the exact
468 evaluation groups and successful A-B-A equality. It does not read targets
or a score mask. `grade_h512_tl3_endpoints.py` rechecks archive, package,
prediction, target-index, split, sequence-order, and source-binding identities
before scoring. It fails closed on changed membership, coefficients, package
code, or checkpoint ancestry.

The output can establish package/native compatibility and local endpoint
behavior on this exposed fold. It cannot establish clean CV, generalization
across folds or seeds, or a public score. Root owns that interpretation and any
later submission decision.

## Pre-execution static corrections

Independent review found that bare run references would package `model.pt` while the protocol binds `model_ep5.pt`. Before any package/replay/score, package commands and member metadata were corrected to explicit `run@ep5` references; canonical run directories remain unchanged. The original TL3 ZIP hash transcription in the prepared plan was also corrected to the pinned source archive hash, and the grader plan digest was updated. No fitted endpoint or outcome was changed.

The [independent static review](../h512_f0_followup_r1/POSTFIT_AND_NATIVE_STATIC_REVIEW_R1.json) passes after these corrections. Prepared commands remain unexecuted; root retains package/replay/grade admission and promotion ownership.

### Native interpreter correction

Before execution, the two replay commands were corrected to `.venv-py311-profile/bin/python`: originalTL3 contains Python3.11native extensions; the root packaging environment is Python3.13. Packaging and target-only grading continue using the root environment. This changes only the serving interpreter, with plan/grader digest repinned; no model/metric/output changed. The fixture already exercised these libraries underPython3.11.

## Conditional CPU execution admitted

The [fixed CPU handoff](run_native_when_ready.py) now waits on the exact live GPU postfit watcher and requires its complete accepted fit/export/score chain. [Execution admission](NATIVE_EXECUTION_ADMISSION_R1.json) pins21source inputs and all new output locations. [Independent static recheck](NATIVE_STATIC_REVIEW_R2.json) passes, including the3.11native interpreter correction; the earlierR1review remains a historical snapshot. [Actual waiter start](NATIVE_WAITER_START_R1.json) records unit `wnn-h512-native-screen-r1-20261021.service`, invocation4d978659258144ca80b409f3ea5ec188, oneCPU/1.8GBRAM/zeroSwap/CUDAhidden and16hcap including waiting. Its [mutable log](native_worker_r1.log) will show two package builds, two complete native replays and the fixed paired grade after upstream success. No step has run yet. No model fit, automatic quality promotion, external upload or submission release is admitted by this worker. Root owns actual evidence review and release selection.

## Recovery binding after launcher page-alignment failure

The original waiter stopped because its original upstream launcher failed before exports. [Recovery admission](NATIVE_EXECUTION_ADMISSION_RECOVERY_R2.json) binds the exact replacement upstream invocation while preserving all fixed endpoint/scoring pins. The separate [native recovery waiter](run_native_recovery_r2.py) waits for the unchanged seven-step GPU sequence, validates its terminal evidence, then packages/replays/grades the original two fixed endpoints. Unit `wnn-h512-native-recovery-r2-20261021.service` is active; [recovery log](native_recovery_r2.log) is mutable. No native evaluation or candidate quality result is claimed at launch.

## October21 R3 recovery completed

The successor cycle's [R3 admission](../../../campaign_2026-10-21/next_cycle_r1/NATIVE_RECOVERY_ADMISSION_R3.json) explicitly binds the repaired NPY-header guard and preserves the exact existing packages. [Terminal sequence report](../../../campaign_2026-10-21/next_cycle_r1/NATIVE_RECOVERY_REPORT_R3.json) completes CP5 replay, SWA345 replay and grading. [Recovered grade](../../../campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json) places both replacements about0.00736 below original TL3 on exposed F0, with essentially no difference from averaging. Historical R2 failures above remain unchanged. The [lineage reconsideration](../../../unified/reviews/H512_ENDPOINT_GAP_INTERIM_REVIEW_20261021_R1.md) records the replaced student's production-training exposure and outstanding precision/contribution diagnostics.
