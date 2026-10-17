---
kind: experiment
status: execution in progress; no endpoint grade yet
created_utc: 2026-10-18
hypothesis_ids: [H-RR17-01]
experiment_ids: [E-RR18-01-LABELS]
evidence_paths:
  - research/campaign_2026-10-18/priority25_execution_r4
  - research/campaign_2026-10-14/cpu_student_frontier_r1
---

# E-RR18-01-LABELS: cold, near-capacity GRU width/depth comparison

This draft continues rank one, H-RR17-01. It references the campaign's frozen recipe rather than copying or changing its training configuration. The proposed scientific comparison is a fresh labels-only H512×1 GRU against a near-parameter H384×2 GRU, evaluated on the four existing source-qualified exposed folds.

The frozen recipe records 2,210,306 versus 2,151,554 trainable parameters: the wide arm has 2.73% more. This is an approximate-capacity architecture comparison. Width and layer count change together, so it does not isolate depth at fixed width or claim exact parameter equality. The historical H512×1/H256×2 screen was more capacity-imbalanced and used a KD/propensity recipe; it does not estimate this cold labels-only contrast.

## Packet files

- [PR.md](PR.md) answers all eleven experiment-draft sections. It records the later decision-threshold addendum separately from the original recipe, which did not freeze a useful-effect threshold before fitting.
- [recipe.json](recipe.json) binds this handoff to the canonical campaign recipe and its execution sources by repository path and SHA. It contains no copied arm configuration; outcome fields remain null.

## Frozen execution and evidence

- [Rank-one recipe](../../campaign_2026-10-18/priority25_execution_r4/RANK01_RECIPE.json) is the canonical scientific configuration (`9292b545fd088df40c4cc54066a745025ee4d30c2bf1574bdb4df341eb16937c`). The [authority](../../campaign_2026-10-18/priority25_execution_r4/AUTHORITY.json) freezes the CP30 matrix, exposed-fold handling, provider limits and current DataSphere budget state.
- [F0 paired plan](../../campaign_2026-10-18/priority25_execution_r4/F0_PAIR_PLAN.json), [exact command manifest](../../campaign_2026-10-18/priority25_execution_r4/RANK01_COMMANDS.json), and [runbook](../../campaign_2026-10-18/priority25_execution_r4/RUNBOOK.md) own the staged fit, resume, export, grade and collection procedure. The independent [endpoint grader](../../campaign_2026-10-18/priority25_execution_r4/code/README.md) binds checkpoint, source labels, masks and exported row order before scoring.
- [Admission review](../../campaign_2026-10-18/priority25_execution_r4/receipts/ADMISSION_REVIEW.json) admitted the first F0/seed1001 pair after source, membership and profile checks. [Profile comparison](../../campaign_2026-10-18/priority25_execution_r4/receipts/PROFILE_COMPARISON.json) records 40 finite FP32 updates per shape on one shared Beleriand GPU. These are engineering profiles, not quality results.
- The [latest copied fit progress](../../campaign_2026-10-18/priority25_execution_r4/receipts/FIRST_FIT_PROGRESS.log) shows the wide F0/seed1001 fit at epoch 0, batch 100/1492. The endpoint checkpoint, exports and independent grade had not completed at that receipt; quality metrics remain null.

## Prior evidence and limits

The [ranked portfolio](../../unified/ideas/PRIORITIZED_25_20261018_R3.md) advanced H-RR17-01 from an exposed one-seed screen. Its reported H512×1 score was 0.67323776 versus 0.66977764 for H256×2, but the historical recipe used a 0.9 teacher-target coefficient and propensity weighting, and the audited parameter-count ratio was 2.27×. Both endpoints were short, exposed-F0 evidence. The [frontier record](../../campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md) explicitly says this is not a converged architecture ranking or quality ceiling.

The current six-epoch fixed endpoint removes teacher/KD, propensity and warm-start ancestry, and narrows the parameter gap. Six epochs still do not establish convergence. All four folds have prior exposure; CP30 cross-fold performance is robustness evidence, not a newly pristine holdout or a public-score forecast. The experiment reports observed results and uncertainty without discarding negative or inconclusive cells.
