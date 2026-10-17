# Cold GRU width versus depth at matched compute

Status: implementation handoff; no new fits. Lineage `student`; `H-DRAFT-GRU-CAPACITY-20261017` / `E-DRAFT-GRU-CAPACITY-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Five one-seed F0 screens completed; H640x1/H512x2/H448x3 unrun. No scaling law established.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Freeze measured compute matching and schedule before quality runs; no CPU model fitting; fresh fullFP32 fold-qualified students.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/train_rnn.py](../../../scripts/train_rnn.py)
- [src/wnn/student_frontier.py](../../../src/wnn/student_frontier.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
