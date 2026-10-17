# Fixed checkpoint trajectory averaging

Status: implementation handoff; no new fits. Lineage `student`; `H-DRAFT-TRAJECTORY-AVERAGE-20261017` / `E-DRAFT-TRAJECTORY-AVERAGE-20261017-R1`. Original fork main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Two F0seed trajectory averages constructed; export/grade pending.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Rebind fixed checkpoint epochs and normalization per fold/seed; preserve existing constructed artifacts; evaluate complete FP32 endpoints.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-13/strategy_execution_r1/trajectory_average_f0_r1/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/train_rnn.py](../../../scripts/train_rnn.py)
- [scripts/predict_run.py](../../../scripts/predict_run.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.

Integration parent: research-team `96b014737211c19d421e29c5595b5ea7e86797ff`. Common catalogues were preserved and this packet appended; no scientific settings or historical ledger rows were dropped.
