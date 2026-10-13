---
kind: experiment
status: admitted
created_utc: 2026-10-12T22:59:14Z
updated_utc: 2026-10-12T22:59:14Z
hypothesis_ids: [H-STRATEGY-20261012-01, H-STRATEGY-20261012-02, H-STRATEGY-20261012-03, H-STRATEGY-20261012-04]
experiment_ids: []
evidence_paths: [CONTRACT.json, COVERAGE.json, FRONTIER.md]
---

# Strategy review execution

## Question or claim

Implement and deeply test every actionable proposal in the [October 12 strategy review](../../unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md). Select complete deterministic predictors using valid local evidence toward top-1 and public score above 0.700. No new quality result exists at campaign admission.

## Mechanism and method

The user approved the implementation plan and authorized all Kaggle allowance and local GPU use without the previous GPU-time or artificial VRAM restrictions. [ADR0034](../../../specs/adr/0034-strategy-execution-full-gpu-allowance.md) records the effective scope and supersedes the experimental stop without rewriting it. The [contract](CONTRACT.json) fixes evaluation, resource and selection policy. The [coverage register](COVERAGE.json) assigns every review proposal a discriminating comparison and tracks missing work. [Implementation plan](PLAN.md) specifies the staged program; [frontier](FRONTIER.md) is the compact current action list.

The first cohort compares target amplitude, causal innovations and gradient credit. Source-qualified F0/F3 and fitting seed 3 are the first screen; substantive family conclusions add seed 4. At most two complete finalists per wave advance to all four folds and two fitting repetitions. Teacher, student and whole-endpoint repetitions are recorded separately.

## Evidence and interpretation

Public C's user-reported score is 0.6671; its all-data weights are not held-out evidence. The clean T3 F0 seed-3/4 CP5 controls score 0.68115536/0.68125842 and share one teacher cohort. The older Z4 F0 reference scores 0.68314181 with different ancestry and precision. These are historical references, not a new matched ranking.

## Limits and validity

Existing folds are exposed; future transfer is tier C. No local delta becomes a public forecast. Every ancestor, normalizer, propensity estimator and learned combiner must exclude evaluated components. The official metric and supplied-data/independent-weight rules remain binding. The train source lacks `is_scored`; mask-aware moment models must declare their approximation.

## Decision and next step

Resume research under the new authority, update GPU guards, prepare and independently review the first actual-producer packet, then fit/export/grade it. Reuse successful checkpoints and collect terminal evidence. Root owns admission and synthesis; each admitted run retains a delivery owner.

## First amplitude packet

[F0 seed-3 target-amplitude packet](target_amplitude_f0_s3_r1/README.md) now has an R3 command-only path correction after R2 P1 failed before fitting: `load_campaign_split` resolved a relative split path against the source-overlay working directory. R3 makes all fit and export split-manifest paths absolute and assigns fresh attempt/output identities; its actual-overlay loader and exact worker-vector source-manifest preflights passed. R3 P1 completed through epoch 1 with a validated `next_epoch=2` state; the root-authorized P2 continuation passed fresh aggregate admission and has reached epoch 2. No amplitude score exists yet. The state/ancestry/schedule checks and exact runtime evidence are bound in [P1_P2_RESUME_GATE_R1.json](target_amplitude_f0_s3_r1/P1_P2_RESUME_GATE_R1.json). F3 remains dependent on a clean T3 teacher cohort, predictions, cache and matched control.

## S14 gradient-credit packet

[F0 seed-3 detach-horizon pair](gradient_credit_f0_s3_r1/README.md) prepares a new matched 500-row/2000-row gradient-credit comparison. Both conditions use micro500, macro/update2000, identical 0.8125 cached target, teacher/parent lineage, auxiliary losses, schedule, denominators, EMA clock and CP5/FP32 export. Existing controlled-loss code already supports the R3 `aux_struct=.3` and `aux_next=.3` terms, confirmed by 14 bounded tests against the immutable source overlay; no trainer edit is needed. The exact 500-row shape still needs a real device profile, root source-manifest recomputation, fresh admission and explicit fit release. This is preparation only; F3 teacher/cache/control inputs remain missing.

The first [S14 profile-only packet R1](gradient_credit_f0_s3_r1/S14_PROFILE_ONLY_COMMANDS_R1.json) is preserved as a malformed, never-launched command: the frozen trainer's real parser rejected the bare `16` in both arms. The [R1 defect receipt](gradient_credit_f0_s3_r1/S14_PROFILE_COMMAND_DEFECT_R1.json) and [R1-to-R2 correction record](gradient_credit_f0_s3_r1/S14_PROFILE_CORRECTION_R1_TO_R2.json) preserve the failure and show no attempt was admitted. R2 corrected its controller vector, but its standalone worker copy remained the malformed R1 vector; the exact mismatch and absent runtime records are documented in the [R2-to-R3 correction receipt](gradient_credit_f0_s3_r1/S14_PROFILE_CORRECTION_R2_TO_R3.json). The sealed [R3 profile-only vectors](gradient_credit_f0_s3_r1/S14_PROFILE_ONLY_COMMANDS_R3.json) use identical controller-suffix and standalone-worker commands, fresh IDs, and a finite 900-second worker timeout. The [R3 actual-parser receipt](gradient_credit_f0_s3_r1/S14_TRAINER_ARGPARSE_PREFLIGHT_R3.json) checks both forms against the frozen overlay before device initialization; the [R3 source preflight](gradient_credit_f0_s3_r1/S14_PROFILE_SOURCE_PREFLIGHT_R3.json) binds the exact Main controller, both 66-file manifests, commands and timeout executable. These are preparation checks only; both actual-device profiles still require root's independent packet review and fresh shared admission. The current [prediction alias workflow R2](PREDICTION_ALIAS_WORKFLOW_R2.json) records the exporter/grader filename mismatch, bounded header-length parsing, actual producer verify-only receipt and oversized-header regression; the previous [workflow R1](PREDICTION_ALIAS_WORKFLOW_R1.json) remains preserved as historical helper-binding evidence.

## Source records

- [Original stop](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json), preserved as historical authority.
- [Final archive](../../unified/archive/2026-10-12/README.md), historical artifacts and exposure.
- [Existing refinement cards](../../unified/ideas/STRATEGY_REFINEMENTS_20261012.json), mechanisms and falsifiers.
- [Initial Kaggle quota](INITIAL_KAGGLE_QUOTA.json), read-only SDK observation at23:04UTC:35,838.966 GPU seconds and72,000 TPU seconds available, zero reserved; the old campaign sub-ceiling is superseded. Refresh before actual admission.
- [Resource/recipe/execution receipts](./), added with exact identities as work completes; private weights remain ignored.
