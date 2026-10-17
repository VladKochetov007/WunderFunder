# Cold GRU width versus depth at matched compute

Status: implementation handoff; no new fits. Lineage `student`; `H-DRAFT-GRU-CAPACITY-20261017` / `E-DRAFT-GRU-CAPACITY-20261017-R1`. Original fork main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Five one-seed F0 screens completed; H640x1/H512x2/H448x3 unrun. No scaling law established.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Freeze measured compute matching and schedule before quality runs; no CPU model fitting; fresh fullFP32 fold-qualified students.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/train_rnn.py](../../../scripts/train_rnn.py)
- [src/wnn/student_frontier.py](../../../src/wnn/student_frontier.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.

Integration parent: research-team `96b014737211c19d421e29c5595b5ea7e86797ff`. Common catalogues were preserved and this packet appended; no scientific settings or historical ledger rows were dropped.

## Shared trainer refactor

Integrated shared trainer source `eb3309a322c1e24b3cc157eaefe3ca0e0290cd42` from research-team `944b4e0c8ce26b842b6b8466aea9f0932bf36ea8`. The [component map](../../../src/wnn/training/README.md) and [engineering receipt](../../unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md) describe reusable data, loss, optimizer, checkpoint and reporting components. Existing CLI options remain available. Static review and lint passed; numerical and throughput equivalence are unmeasured. This integration adds no scientific result or launch authority.
