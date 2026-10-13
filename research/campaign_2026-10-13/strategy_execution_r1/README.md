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

The user subsequently requested merging `claude/stoic-euler-70wff3` and considering its [first-principles critique](../../unified/reviews/FIRST_PRINCIPLES_REVIEW_PLAN_20261012.md). Main merge `20c904da` preserves that proposal. The [experiment assessment](../../unified/ideas/FIRST_PRINCIPLES_EXPERIMENTS_20261013.md) adds S36–S39 (compact ensembles, isolated averaging, symmetry diagnosis and causal cross-fitting), extends propensity power to include zero, and qualifies the review's historical evidence. These additions do not replace the initial cohort or adopt its proposed blanket stop list.

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

The first [S14 profile-only packet R1](gradient_credit_f0_s3_r1/S14_PROFILE_ONLY_COMMANDS_R1.json) is preserved as a malformed, never-launched command: the frozen trainer's real parser rejected the bare `16` in both arms. The [R1 defect receipt](gradient_credit_f0_s3_r1/S14_PROFILE_COMMAND_DEFECT_R1.json) and [R1-to-R2 correction record](gradient_credit_f0_s3_r1/S14_PROFILE_CORRECTION_R1_TO_R2.json) preserve the failure and show no attempt was admitted. Corrected [R2 profile-only vectors](gradient_credit_f0_s3_r1/S14_PROFILE_ONLY_COMMANDS_R2.json) pass the exact overlay trainer's parser before device initialization; fresh 66-file source manifests and command hashes are in the [R2 source preflight](gradient_credit_f0_s3_r1/S14_PROFILE_SOURCE_PREFLIGHT_R2.json). Both R2 device profiles remain unauthorized pending root's independent source check and fresh shared admission. The current [prediction alias workflow R2](PREDICTION_ALIAS_WORKFLOW_R2.json) records the exporter/grader filename mismatch, manual header-length bound, actual producer verify-only receipt and oversized-header regression; the previous [workflow R1](PREDICTION_ALIAS_WORKFLOW_R1.json) remains preserved as historical helper-binding evidence.

## Source records

- [Matched innovation pair](innovation_f0_s3_r1/README.md) binds two new Main-derived F0 runs with identical80-channel width and a zero-innovation control. Real parent expansion preserves the initial function; actual fold selection and six selector tests pass. Independent packet review and GPU profiles precede fitting; root owns delivery.
- [Fixed two-seed trajectory averages](trajectory_average_f0_r1/README.md) materialize S37's predeclared epoch-3/4/5 averages from real clean F0 checkpoints. Actual producer checks and three correctness tests pass; independent review, GPU prediction and paired grading remain pending. The new artifacts add no fitting or serving cost, but have no measured quality yet.
- [F3 seed-3 root release](first_cohort_f3_teachers_r2/ROOT_RELEASE_R2.json) independently checks the sealed R2 source, nested mounts, exact bootstrap, quota window and output accounting before provider dispatch. Three packet tests pass; this authorizes one conditional profile/fit/export/collection tranche, not a model-quality claim. The original fractional-second quota threshold is corrected by the delivery addendum to 35,838 seconds, preserving the same 17,019-second two-device window.
- [Historical auxiliary/loss audit](HISTORICAL_AUXILIARY_AUDIT_R1.md) and [terminal-history receipt](HISTORICAL_AUXILIARY_AUDIT_R1.json) complete S25's evidence-location task. Seven completed histories and scoped historical negatives are preserved; no new quality computation or broad family rejection.
- [Original stop](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json), preserved as historical authority.
- [Final archive](../../unified/archive/2026-10-12/README.md), historical artifacts and exposure.
- [Existing refinement cards](../../unified/ideas/STRATEGY_REFINEMENTS_20261012.json), mechanisms and falsifiers.
- [Initial Kaggle quota](INITIAL_KAGGLE_QUOTA.json), read-only SDK observation at23:04UTC:35,838.966 GPU seconds and72,000 TPU seconds available, zero reserved; the old campaign sub-ceiling is superseded. Refresh before actual admission.
- [Independent reviews](reviews/README.md) explain exact code/recipe checks and limitations. Resource R1 confirms GPU-policy behavior and identifies an unused RAM-reservation gap for correction before concurrent admission.
- [S37 trajectory-average review](reviews/trajectory_average_f0_r1/README.md) independently verifies both materialized averages against actual checkpoints and continuation states and confirms strict CPU loading through the staged predictor. Export and paired grading remain pending; no quality result is claimed.
- [Evaluator precheck](EVALUATOR_PRECHECK_R1.json) binds scorer/evaluation source hashes and 43 passing correctness tests before any new candidate outcomes. This verifies known metric, mask and cross-fitting cases, not model improvement.
- [Domain-cache schema observation](DOMAIN_CACHE_SCHEMA_OBSERVATION_R1.json) records bounded actual-producer metadata: existing train summaries cover only 2,000 sequences and mix observable and target-derived fields. Stress-panel construction must whitelist observable fields and audit source/group coverage; this is a preparation finding, not a domain-shift result.
- [Initial domain panels](DOMAIN_PANELS_R1.md) explain the [frozen four-fold definitions](DOMAIN_PANELS_R1.json), three input-only descriptors, component-frequency stress mixtures and three passing correctness tests. Every stratum has at least 60 components. [Independent review](reviews/domain_stress_r1/README.md) prompted a shared-mask evaluator correction; three other panel families remain pending. No candidate result is claimed.
- [Innovation implementation and full-path correction](INNOVATION_IMPLEMENTATION_R1.md) records S08/S09 training/streaming/export integration, matched-width controls and the long-sequence numerical failure/fix. The corrected real 20,000-row fixture passes the unchanged tolerance; the [independent implementation review](reviews/innovation_implementation_r1/README.md) confirms CPU batch, Torch streaming and ONNX parity on the checked fixture. GPU profiling and model-quality experiments remain pending.
- [S38/H-317 symmetry diagnostic](s38_mirror_symmetry_r1/README.md) contains the frozen map proposals, actual 32-component feature-only input measurements and execution receipt. The exact side-swap marginal screen fails; reflection and sign-only non-crossings are not preservation passes. Target semantics remain unknown, no augmentation is authorized, and the recorded 400 MB metadata-scope OOM is not a scientific result.
- [Shared resource resume](RESOURCE_RESUME_R1.json) records the supported reopening of the original runtime and GPU lock after [independent R2 review](reviews/resource_r2/README.md). The old generic queue remains stopped. Overlapping support uses the full active reservation in its conservative memory check because cgroup observations are not atomic; this receipt authorizes no unreviewed recipe.
- [Resource/recipe/execution receipts](./), added with exact identities as work completes; private weights remain ignored.
- [Initial amplitude release](target_amplitude_f0_s3_r1/ROOT_RELEASE_R1.json) records the exact R2 contract and conditional stage authority. The [independent R2 review](reviews/amplitude_recipe_r2/README.md) links its actual pre-fit path failure and the required command correction. This was an execution failure with no optimizer updates or model-quality evidence; immutable R2 receipts remain preserved.
