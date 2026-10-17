# Expand clean heterogeneous recurrent teacher cohort

Status: implementation handoff; no new fits. Lineage `teacher/connection`; `H-DRAFT-TEACHER-COHORT-20261017` / `E-DRAFT-TEACHER-COHORT-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Single cleanF3 teacher and FP32 exports are complete; broader cohort incomplete.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Do not redo completed export-only task; rebuild fullFP32 clean teacher cohorts for each evaluated fold, document every teacher fitting membership.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-15/f3_teacher_export_r2/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/train_rnn.py](../../../scripts/train_rnn.py)
- [scripts/build_distill_cache.py](../../../scripts/build_distill_cache.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
