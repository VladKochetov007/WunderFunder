---
kind: review
status: complete
created_utc: 2026-10-20T20:16:08Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-20/submission_delivery_r1/SAVED_INPUTS.json
---

# Saved models and local continuation: independent input audit

## Question and scope

Can completed FG19 models or existing local assets support a new submission?
Reviewer: native `gpt-6-luna`, maximum reasoning, role `overnight_evidence_audit`.
This was source/JSON inspection, without loading tensors, predictions or targets.

## Findings

The [saved-model packet](../../campaign_2026-10-20/submission_delivery_r1/README.md)
pins FG19 control/EMA at epoch5 and their equal output mixture. Both remote fits
report 44,760 successful strict-FP32 updates and accepted tensor integrity.
Neither has a quality score. Their 467-input representation must be restored
before prediction; this machine currently lacks those weights. Control
acceptance omits the split field, but its admission pins split SHA `c09f7240…`.
Stale pending-stop fields in acceptance records are superseded by the originating
campaign's exact-stop receipts.

Local alternatives use split SHA `09263ad4…` and different teacher cache bytes.
Matching counts of 468 evaluation groups do not establish identical membership.
Do not substitute local assets and describe the result as a replay of FG19.

The local clean-T3 students are H256×2, despite the target cache's H390 name.
Seed3/4 CP5 scores are 0.68115536/0.68125842 on exposed F0. Their direct fitting
membership excludes evaluation groups; initializers and teacher ancestry include
FP16 fitting. The target cache stores `0.8125 * mean(three epoch2 predictions)`,
used with a 0.9 teacher / 0.1 label blend. Cache scale1 preserves that contract.
These are historical results from the
[teacher/student review](CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md).

The stronger old-Z4 student remains relevant: 0.68314181 on this panel. Its
previous equal blend with the teacher trio scored 0.68316743, delta +0.00002562
with conditional component interval [-0.00138485,+0.00139836]. That result does
not establish a useful gain; four models also increase serving cost.

## Objective fidelity and correction

Producer `68ff682977b822f38351aac0f621146f5f13b440` implements NLL in
`src/wnn/training/losses.py` using the same clipped teacher/label target as WMSE.
The CLI still requires `--loss wmse`; `--mean-variance-objective nll` selects the
actual main NLL term. Both arms can share two zero-initialized variance outputs.

The review initially overstated that NLL requires the 467-input transform.
The reviewer corrected this after inspecting setup: omitting
`--representation-mode` retains 211 inputs; the variance head changes outputs
only. A local NLL-only pair therefore requires no missing representation asset.
This correction precedes any profile or fit admission.

## Serving source review

The isolated native-builder diff has the correct 467-slot layout, raw signed
delta EMA, first-row initialization, explicit mode checks and state reset.
Legacy 211/323 layouts remain explicit; sparse modes fail closed. Appended
normalizers must be zero/one. No concrete layout bug was found statically.
Producer block-cumsum and native sequential EMA have different FP32 operation
orders. Actual feature/output parity remains necessary; static inspection cannot
establish it. The builder's synthetic feature comparison does not establish
actual checkpoint prediction parity.

## Decision and limits

Evaluate FG19 when accessible. Prepare a separate matched local FP32 WMSE/NLL
continuation using existing assets and explicit historical AMP ancestry. The
smaller likelihood pilot motivates this experiment; it establishes no GRU gain.
One F0 pair is directional evidence; another seed still shares the teacher cohort
and exposed fold. No public forecast or ready ZIP follows from this audit.


## Profile admission review

The same independent critic reviewed the actual profile log and input-verification receipt and recommended admitting the unchanged pair. Each profile completed60/60updates with finite losses and complete preclip gradient statistics, no AMP skips and returncode0. The four previously unresolved propensity digests are bound in actual verification. This is a loss/gradient gate; final model/optimizer-state finiteness was not measured. Generic profile output omitted CUDA peak statistics; outer timing reports3,491,184,640bytes peakRSS, as recorded in the [profile result](../../campaign_2026-10-20/submission_delivery_r1/nll_adaptation/profile_result.json).

The NLL profile's100%clip rate and0.176mean clip factor are retained as a diagnostic. The objective treatment includes its interaction with common global clipping. The critic advised keeping the frozen LR/clip and observing epoch diagnostics, rather than silently retuning this arm. Root admitted the pair under the unchanged aggregate resource boundary.
