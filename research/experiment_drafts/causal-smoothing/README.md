# Past-only smoothing while retaining raw inputs

Status: implementation handoff; no new fits. Lineage `teacher/student`; `H-DRAFT-CAUSAL-SMOOTHING-20261017` / `E-DRAFT-CAUSAL-SMOOTHING-20261017-R1`. Original fork main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. User proposal; causal helper only, no fit.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Wire feature/cache/streaming path; freeze smoothing coefficient using fitting-only development; no centered smoothing or whole-sequence fitted statistics.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [src/wnn/causal_research_features.py](../../../src/wnn/causal_research_features.py)
- [scripts/train_rnn.py](../../../scripts/train_rnn.py)

New code in this PR: src/wnn/causal_research_features.py Shared primitive files have identical contents across related draft PRs, so they should merge once; they are not independent replications. Circle arithmetic is strictFP32, adapted from the source actor commit bc0ead82; cancellation and geometric validity remain untested.

Integration parent: research-team `96b014737211c19d421e29c5595b5ea7e86797ff`. Common catalogues were preserved and this packet appended; no scientific settings or historical ledger rows were dropped.
