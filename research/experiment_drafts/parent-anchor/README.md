# Frozen-parent output anchor

Status: implementation handoff; no new fits. Lineage `connection`; `H-DRAFT-PARENT-ANCHOR-20261017` / `E-DRAFT-PARENT-ANCHOR-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. S resource profile completed; no S fullfit. Endpoint mixture result is not S evidence.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Re-export parent in verifiedFP32; separate calibration from teacher-function change; bind fold-clean anchors.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-15/student_quality_tradeoff_r1/frozen_parent_anchor_r1/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/train_rnn.py](../../../scripts/train_rnn.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
