# Per-field ages versus seven group clocks

Status: implementation handoff; no new fits. Lineage `student/connection`; `H-DRAFT-FIELD-AGE-20261017` / `E-DRAFT-FIELD-AGE-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. F only profiled50 updates; C/G fitted. G-C +0.00002956, conditional interval crosses0.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. FP32 successor of F/G; regenerate fold-clean teacher/cache and parent bindings; retain rank/optimizer-geometry confound controls.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-15/gru_parent_growth_r1/README.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [src/wnn/research_features.py](../../../src/wnn/research_features.py)
- [scripts/train_rnn.py](../../../scripts/train_rnn.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
