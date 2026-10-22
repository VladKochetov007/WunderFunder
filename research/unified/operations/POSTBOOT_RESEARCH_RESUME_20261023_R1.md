---
kind: experiment
status: running
created_utc: 2026-10-22T21:23:04Z
updated_utc: 2026-10-22T21:23:04Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-ONLINE-RELATIONSHIPS-20261022-R1]
evidence_paths: [research/campaign_2026-10-21/next_cycle_r1/POSTBOOT_RESUME_R1.json]
---

# Postboot research resume and new cost evidence

## Question or claim

The user explicitly resumes continuous research and invokes conduct-creative-research and research-reframe. This supersedes the reboot pause while retaining the original 30-hour allocation, stage ceilings, no paid compute/upload and 12-hour production reserve. [Contract](../../campaign_2026-10-21/next_cycle_r1/REFRAME_CONTRACT_R1.json) separates the fixed runnable screen from fresh hypotheses with zero additional fit authority.

## Mechanism and method

Postboot inspection verifies physical Btrfs `/tmp`, physical SSD scratch, 16GiB SSD swap at priority200 with zero use, and no zram. GPU entry56°C/26MiB/0% utilization; host44GiB available. The research cap reverted to20GB because the older correction was runtime-only. Before starting workloads, root persistently sets two CPUs, MemoryHigh4.4GB, MemoryMax5GB, MemorySwapMax0 and verifies after daemon-reload. The [resume receipt](../../campaign_2026-10-21/next_cycle_r1/POSTBOOT_RESUME_R1.json) pins the setting and state. Historical observers remain disabled; each admitted worker has its own thermal guard.

## Evidence and interpretation

The reviewed B8 parent-cache implementation and one-time warmup accounting were profiled on16complete fit and8complete evaluation sequences per fold, all12disposable heads/three epochs. [Execution/accounting](../../experiment_drafts/online-relationships-20261022/POSTBOOT_PROFILE_EXECUTION_R1.json) records71.032service seconds,119.705CPU seconds,2.7G journalpeak, zero observed task swap/OOM and no cooling pauses. Complete measured phases project4,755.379seconds: parentcache2,255.665, fits2,201.192, evaluation200.036, normalization33.461 and exports/startup64. 

This is cost evidence only. Fitting labels were used for the disposable profile; outer evaluation labels/masks were not read. No scientific head was admitted and no official quality result exists. The [full packet](../../experiment_drafts/online-relationships-20261022/README.md) preserves preceding failed attempts, exact source review and tests. Independent cost review must include cold full-cache reads, repeated hashes/validation, grading/provenance and thermal margin before admission.

## Limits and validity

All local folds remain exposed, two cells confound fold/seed/parent, and fit-side frozen-parent predictions are in-sample. Small sample caches fit RAM, whereas full caches total31.86GB; extrapolated readout cost omits cold disk behavior. Disk data has100.41GiB available but Btrfs metadata is20.42/21GiB; do not create more worktrees. [Fresh Kaggle quota](../../campaign_2026-10-21/next_cycle_r1/KAGGLE_QUOTA_POSTBOOT_R1.json) reports about0.45GPUhours, rounded; no provider launch admitted. New source-grounded ideas get separate cards, not automatic fits.

## Decision and next step

Root owns admission, delivery actor owns the complete fixed panel, and an independent grader owns target outcomes. Preserve the completed warm F0 CP1/full CPU predictions. The remaining three warm fits require a safe complete-remainder cost gate within16,235.769seconds, and are not silently truncated or changed. Continue within the existing allocation until explicit stop or a documented spending/validity gate requires a new decision.

## Source records

[Resume](../../campaign_2026-10-21/next_cycle_r1/POSTBOOT_RESUME_R1.json), [contract](../../campaign_2026-10-21/next_cycle_r1/REFRAME_CONTRACT_R1.json), [frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md), [online packet](../../experiment_drafts/online-relationships-20261022/README.md), [reboot pause history](REBOOT_PAUSE_20261022_R2.md).
