---
kind: review
status: complete
created_utc: 2026-11-14T15:09:00Z
updated_utc: 2026-11-14T15:09:00Z
hypothesis_ids: [H-Q14-PR99-STRUCTURED-PERTURBATIONS]
experiment_ids: []
evidence_paths: [PR99_SNAPSHOT_R1.json]
---

# New team PR99: causal structured perturbations

## Question or claim

The final GitHub refresh found a new [PR99](https://github.com/VladKochetov007/WunderFunder/pull/99) by Ivanhacker13, head `320ebd6ca38cf5cd16a708dd1a088c5f06f23e48`. This supersedes the earlier scout's no-new-push observation at its own snapshot. It is prepared implementation and a frozen experiment design, with zero fits, numerical model tests, profiles or measured quality gains.

## Mechanism and method

Two ideas target redundant causal observations. Stale i1 blocks hold the last observed same-sequence i1 values while preserving i0 and auxiliary channels; equal-count independent row omissions provide a control. Historical-key dropout removes old attention keys in bursts or independent positions while protecting the current chunk and32 recent cached keys. Evaluation disables both interventions. Burst and IID arms match selected counts, not stale age or removed attention mass; correlated auxiliary information may remain.

The panel uses label-only Transformer24×768,12heads, context1024, chunk/gradient horizon256, two B300s and BF16/BF16x9. Clean, stale-IID, stale-burst, history-IID, history-burst each use seeds208401/208402:6 input-screen fits plus4 history fits reusing2 exact clean anchors,10 total. Fitting is10,325 train-source sequences with zero valid; evaluation is the exposed1,873-sequence valid block. Fixed E4/3280 updates, no epoch selection or KD. A mean burst−clean gain≥.001 with both positive motivates confirmation; burst−IID must also be positive to support the structural explanation.

## Evidence and interpretation

Read the frozen PR body/file list and [runtime design at the exact head](https://github.com/VladKochetov007/WunderFunder/blob/320ebd6ca38cf5cd16a708dd1a088c5f06f23e48/research/experiment_drafts/structured-perturbations-b300-20261114/RUNTIME_DESIGN.md). The author reports AST/Ruff checks only. No checkpoint, prediction or score can improve today's package yet. This is a different hypothesis from the already tested unstructured GRU column-view masks; those negative results do not test temporal staleness or attention-history blocks.

## Limits and validity

No implementation execution or independent numerical verification was performed by this readout. Perturbations can destroy useful freshness/long-memory signal. Two fitting seeds on one exposed split cannot establish unseen transfer. The BF16/B300 recipe differs from the root campaign's strict FP32 and local/Kaggle allocation; publication supplies no root compute authorization.

## Decision and next step

Keep as a source-grounded, untested direction. Reuse audited implementation only in a separately frozen matched recipe if allocated later; no B300 launch, BF16 fit or automatic import is admitted here. The available measured FBC/TLead teacher-target gain remains the current prepared distillation priority.

## Source records

[Exact PR snapshot](PR99_SNAPSHOT_R1.json), [earlier PR scout](PR_SCOUT_R1.md), [teacher target result](../teacher_pool_r3/TWO_TEACHER_RESULT_R1.md).
