---
kind: idea
status: proposed
created_utc: 2026-10-17T12:15:27Z
updated_utc: 2026-10-17T12:15:27Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/unified/operations/RESEARCH_PAUSE_20261015_R1.md
---

# Research-team draft inventory

## Scope and evidence

This delivery prepares one local branch per listed idea, based on the October15 interrupted campaign, explicitly reused October13 continuations and October17 chat proposals. It is not a claim that every historical hypothesis in the repository has been exhausted. The independent read-only inventory checked campaign receipts; completed C/G fits and the cleanF3 export remain completed sources. No fits or provider jobs result from this preparation. Publication was subsequently authorized; fourteen draft PRs are now open targeting research-team. Existing evaluation folds are exposed.

Read the [workflow](../methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) and [original AR teacher note](CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md). Each local `experiment/<slug>` branch contains `research/experiment_drafts/<slug>/README.md`, English `PR.md`, `recipe.json` and `run_plan.py`; implementation sources are referenced rather than copied. `run_plan.py` prints the paired matrix; it does not train. Missing data/GPU integration is explicit per packet. Engineering TPU parity is not a30-run quality experiment.

| Local branch | Lineage | Actual evidence / remaining state |
|---|---|---|
| `experiment/field-age` | student/connection | **Per-field ages versus seven group clocks.** F only profiled50 updates; C/G fitted. G-C +0.00002956, conditional interval crosses0. [Source](../../../research/campaign_2026-10-15/gru_parent_growth_r1/README.md). |
| `experiment/gru-growth` | student | **Function-preserving GRU widening.** W50-update profile and initializer parity completed; no W fullfit. [Source](../../../research/campaign_2026-10-15/gru_parent_growth_r1/README.md). |
| `experiment/parent-anchor` | connection | **Frozen-parent output anchor.** S resource profile completed; no S fullfit. Endpoint mixture result is not S evidence. [Source](../../../research/campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/README.md). |
| `experiment/initial-state` | student | **Learned causal initial-state family.** Static FP32 subset recipes only; preflight/profile/fits unrun. [Source](../../../research/campaign_2026-10-15/initializer_family_f0_r1/README.md). |
| `experiment/gru-capacity` | student | **Cold GRU width versus depth at matched compute.** Five one-seed F0 screens completed; H640x1/H512x2/H448x3 unrun. No scaling law established. [Source](../../../research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/README.md). |
| `experiment/transformer-ar` | teacher | **Causal Transformer with output autoregression.** Detailed architecture proposal; no teacher fit or B300 profile. [Source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md). |
| `experiment/input-encoding` | teacher | **Grouped feature encoding for causal Transformer.** User proposal; grouped model starter only, no measured result. [Source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md). |
| `experiment/causal-smoothing` | teacher/student | **Past-only smoothing while retaining raw inputs.** User proposal; causal helper only, no fit. [Source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md). |
| `experiment/circle-feature` | teacher/student | **Trailing geometry and circle residual feature.** User idea ±epsilon; coordinate interpretation remains unverified. Diagnostic helper is not evidence. [Source](../../../docs/data_overview.md). |
| `experiment/teacher-family` | connection | **GRU versus Transformer teachers for the same GRU student.** User proposal; historical successful KD does not establish Transformer ancestry. [Source](../../../research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md). |
| `experiment/pretrained-teacher` | teacher/connection | **Permitted Chronos or TimesFM teacher transfer.** Unimplemented proposal; exact pretrained checkpoints/license/input semantics unresolved. [Source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md). |
| `experiment/teacher-cohort` | teacher/connection | **Expand clean heterogeneous recurrent teacher cohort.** Single cleanF3 teacher and FP32 exports are complete; broader cohort incomplete. [Source](../../../research/campaign_2026-10-15/f3_teacher_export_r2/README.md). |
| `experiment/trajectory-average` | student | **Fixed checkpoint trajectory averaging.** Two F0seed trajectory averages constructed; export/grade pending. [Source](../../../research/campaign_2026-10-13/strategy_execution_r1/trajectory_average_f0_r1/README.md). |
| `experiment/causal-innovation` | teacher/student | **Causal innovation residuals beyond moment features.** Existing matched pair blocked on cache schema before fits; execution failure is not negative science. [Source](../../../research/campaign_2026-10-13/strategy_execution_r1/innovation_f0_s3_r1/README.md). |
| `experiment/tpu-parity` | engineering | **Resolve native versus TPU full-FP32 parity.** OneTPU update/export ran; reset passed, native parity failed max0.00777 versus1e-4. Cause unresolved. [Source](../../../research/campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md). |

## Execution and decision

The research-team branch carries a local branch/commit catalogue after preparation. Actual PR URLs and source SHAs are in the [published catalogue](../../experiment_drafts/README.md); use `git show experiment/<slug>:research/experiment_drafts/<slug>/PR.md` to read a local draft. No leaderboard deltas are predicted for unrun ideas. Historical deltas are conditional local evidence, not transferable additive public gains. Each packet must freeze a useful-effect threshold and exact source-clean cell bindings before running; the current code handoff is not execution approval.

## Source records

- [Pause and terminal closeout](../operations/RESEARCH_PAUSE_20261015_R1.md): C/G complete, new F/W/S and initializer fits unrun.
- [Warm mechanism campaign](../../campaign_2026-10-15/gru_parent_growth_r1/README.md): source, profiles, growth parity and recipe versions.
- [B300 skill search](../sources/B300_AGENT_SKILLS_REVIEW_20261017_R1.md): inspected sources and adaptation limits.

## Publication receipt

GitHub API confirmed14 open draft PRs with base research-team and matching source head SHAs. The [catalogue](../../experiment_drafts/README.md) links every PR and its [machine manifest](../../experiment_drafts/CATALOG_20261017_R1.json). No experiment code is promoted into main by this index update, and no fits were launched.
