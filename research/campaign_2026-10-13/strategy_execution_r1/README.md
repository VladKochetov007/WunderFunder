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

## Source records

- [Original stop](../../campaign_2026-10-12/STOP_NEW_RESEARCH_AND_FINAL_ARCHIVE_20261012_R1.json), preserved as historical authority.
- [Final archive](../../unified/archive/2026-10-12/README.md), historical artifacts and exposure.
- [Existing refinement cards](../../unified/ideas/STRATEGY_REFINEMENTS_20261012.json), mechanisms and falsifiers.
- [Initial Kaggle quota](INITIAL_KAGGLE_QUOTA.json), read-only SDK observation at23:04UTC:35,838.966 GPU seconds and72,000 TPU seconds available, zero reserved; the old campaign sub-ceiling is superseded. Refresh before actual admission.
- [Independent reviews](reviews/README.md) explain exact code/recipe checks and limitations. Resource R1 confirms GPU-policy behavior and identifies an unused RAM-reservation gap for correction before concurrent admission.
- [Evaluator precheck](EVALUATOR_PRECHECK_R1.json) binds scorer/evaluation source hashes and 43 passing correctness tests before any new candidate outcomes. This verifies known metric, mask and cross-fitting cases, not model improvement.
- [Domain-cache schema observation](DOMAIN_CACHE_SCHEMA_OBSERVATION_R1.json) records bounded actual-producer metadata: existing train summaries cover only 2,000 sequences and mix observable and target-derived fields. Stress-panel construction must whitelist observable fields and audit source/group coverage; this is a preparation finding, not a domain-shift result.
- [Initial domain panels](DOMAIN_PANELS_R1.md) explain the [frozen four-fold definitions](DOMAIN_PANELS_R1.json), three input-only descriptors, component-frequency stress mixtures and three passing correctness tests. Every stratum has at least 60 components; three other panel families and independent review remain pending. No candidate result is claimed.
- [Shared resource resume](RESOURCE_RESUME_R1.json) records the supported reopening of the original runtime and GPU lock after [independent R2 review](reviews/resource_r2/README.md). The old generic queue remains stopped. Overlapping support uses the full active reservation in its conservative memory check because cgroup observations are not atomic; this receipt authorizes no unreviewed recipe.
- [Resource/recipe/execution receipts](./), added with exact identities as work completes; private weights remain ignored.
