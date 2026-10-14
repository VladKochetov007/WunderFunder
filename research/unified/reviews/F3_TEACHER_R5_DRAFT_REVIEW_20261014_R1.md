---
kind: review
status: complete
created_utc: 2026-10-14T01:31:33Z
updated_utc: 2026-10-14T01:31:33Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/DRAFT_STATUS_R5.md
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/ROOT_DRAFT_INTEGRITY_R5.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/KAGGLE_QUOTA_GUARD_R5.json
---

# Clean F3 teacher R5 draft-packet review

## Question or claim

Does the offline R5 packet preserve the frozen F3 seed-3 teacher recipe while bounding cloud inputs, process time and checkpoint collection, and does it remain closed to dispatch until a fresh quota/resource admission?

## Mechanism and method

Reviewed the draft packet in `research/campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/`. The R5 job's `argv`, environment, inputs, limits, artifacts and kind match the R4 fit after only the run-name suffix change. The job remains one scratch H384×2, three-epoch, seed-3, fold-3 Muon-all fit with the frozen auxiliary losses; it packages no export worker. The pinned 200-update T4 profile is reused strictly as capacity evidence, not a fit or quality result.

## Evidence and interpretation

The archive hash matches both the bundle manifest and bootstrap constant. Its 83 entries are the 82 individually pinned code files plus the job spec; all code hashes match archived bytes. The 13 external-mount declarations match the bootstrap map and job input paths. Only `train.parquet` and `valid.parquet` are deferred: the runner requires their mounted symlink and exact size, then the frozen worker checks the portable registry and streams full content hashes against the fold manifest before fitting. Other declared inputs receive the runner's full local hash check. The offline bootstrap fixture uses small fake mount files and intercepts `Popen`, so it verifies dispatch wiring and cleanup only; it does not read live Kaggle mounts or execute a worker.

The process cap is 10,000 seconds: 8,500 for the fit, 900 for finalization and up to 600 for bootstrap/profile checks. Quota arithmetic conservatively assumes two billable GPUs per wall-second and keeps 1,800 GPU-seconds as reserve. The 9,400-second stage budget is checked after bootstrap and profile reuse; it is not a separate 10,000-second provider guarantee. Output collection requires a bounded inventory, at least 4 GB free physical disk, no overwrite and one download attempt. Checkpoint/provenance/training-state collection is separated from the three pending prediction exports. Their per-seed raw payload geometry sums correctly to 2.2048 GB, with fit-train, fit-valid and full-valid kept distinct.

## Limits and validity

The quota snapshot is intentionally stale and the packet is not launchable. No Kaggle write, live mount validation or R5 fit occurred. A clean seed-3 checkpoint alone is not the planned three-teacher cohort and cannot be called a completed cache without all required prediction arrays and sidecars. The reused profile's terminal kernel status was `ERROR` after a completed profile result; the receipt preserves that distinction and does not turn it into fit evidence.

## Decision and next step

PASS for offline source, recipe and containment review; HOLD for dispatch until QAT R4 is terminal, root has a fresh quota snapshot and no active owned kernel, and current aggregate host/GPU admission passes. The reviewed bundle itself performs no billing change, retry or provider write. If root refreshes any sealed quota/input fields, rebind and recheck the archive/bootstrap hashes before release.

## Source records

- [R5 draft status](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/DRAFT_STATUS_R5.md)
- [R5 packet README and recipe context](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/README.md)
- [R5 root integrity receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/ROOT_DRAFT_INTEGRITY_R5.json)
- [R5 actual bootstrap fixture receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/OFFLINE_BOOTSTRAP_TEST_CURRENT_DRAFT_R5.xml)
- [R5 bounded collection plan](../../campaign_2026-10-14/cpu_student_frontier_r1/f3_teacher_fit_r5/OUTPUT_COLLECTION_PLAN_R5.json)
