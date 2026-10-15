---
kind: incident
status: running
created_utc: 2026-10-15T17:55:39Z
updated_utc: 2026-10-15T17:55:39Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json
---

# User-requested research pause and bounded drain

## Question or claim

The latest user instruction is “pause research when its convenient”. It supersedes continuous research authorization and earlier conditional admissions. New research admission is closed immediately; healthy admitted work may reach a useful terminal checkpoint.

## Mechanism and method

Only `warm-mechanism-fit-g-20261015-r1` is currently fitting. Its owner `cpu_inference` may finish this already-running two-epoch fit, validate its trusted CP1/state, and execute its previously owned full-valid FP32 export. `independent_review` may independently grade those predictions with the existing official evaluator against C. These are the complete drain set; no F, W, S, initializer fit/profile, additional diagnostics or cloud retry may start. Keep the aggregate two-core, one-GPU, physical-scratch and memory limits unchanged.

The global controller barrier must be closed after the exact G export is admitted, so that it cannot inadvertently block this authorized collection step. The owner must preserve the barrier and report terminal units, checkpoint/prediction identities and grade. No generic research queue or provider retry is authorized in the interim.

## Evidence and interpretation

C is complete and independently graded: official exposed-F0 WP 0.6812265132 versus original parent 0.6809412068; paired component interval crosses zero. G was observed healthy at epoch0 batch60/746; approximately20–25 minutes of fitting remained at the observed cadence, followed by export and grading. This is an estimate, not a completion claim.

The Kaggle R5 version3 cold-shape profiles are terminal ERROR with zero optimizer updates. The strict cache-root validator rejects external symlink targets. No cloud job remains in that lane. Preserve the diagnosed successor proposal—stream-copy the two hash-bound arrays into their registered physical cache root, retain strict containment, test the actual cache consumer—but do not implement or dispatch it during pause.

The initializer actor may seal existing code, static findings and a runnable resume handoff, without a new profile or fit. The source-resolution correction must explicitly bind initializer modules to their pinned source tree. Static inspection is not fit or quality evidence.

## Limits and validity

Previously exposed folds remain exposed. No new architecture winner, teacher ceiling, public-score gain or CPU-policy speedup is established by this pause. CPU A/B/C/D ZIPs remain ready for user-owned uploads; no submission is uploaded by an agent.

## Decision and next step

Pause all research after the bounded drain; only explicit user resume reopens admission. Preserve all frozen sources, recipes, negative results and unrun arms. Update this record with the actual drain outcome rather than silently treating the estimate as completion.

## Source records

- [C/G/F/W delivery](../../campaign_2026-10-15/gru_parent_growth_r1/README.md).
- [Student-quality evidence and S proposal](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md).
- [CPU submission ZIP handoff](../../../submissions/10-14-cpu-policy-r1/README.md).
- [Cloud shape-pair packet](../../campaign_2026-10-15/student_shape_pair_f0_r1/README.md).
