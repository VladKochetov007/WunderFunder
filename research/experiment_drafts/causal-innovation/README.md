# Causal innovation residuals beyond moment features

Status: implementation handoff; no new fits. Lineage `teacher/student`; `H-DRAFT-CAUSAL-INNOVATION-20261017` / `E-DRAFT-CAUSAL-INNOVATION-20261017-R1`. Original fork main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Existing matched pair blocked on cache schema before fits; execution failure is not negative science.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Repair producer-consumer homogeneous cache contract using actual fixtures; bind fold-clean inputs andFP32 successor recipes.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-13/strategy_execution_r1/innovation_f0_s3_r1/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [src/wnn/research_features.py](../../../src/wnn/research_features.py)
- [scripts/train_rnn.py](../../../scripts/train_rnn.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.

Integration parent: research-team `96b014737211c19d421e29c5595b5ea7e86797ff`. Common catalogues were preserved and this packet appended; no scientific settings or historical ledger rows were dropped.
