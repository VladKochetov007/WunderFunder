# Trailing geometry and circle residual feature

Status: implementation handoff; no new fits. Lineage `teacher/student`; `H-DRAFT-CIRCLE-FEATURE-20261017` / `E-DRAFT-CIRCLE-FEATURE-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. User idea ±epsilon; coordinate interpretation remains unverified. Diagnostic helper is not evidence.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Declare coordinate pairs, trailing window and epsilon from fitting data; include condition-number/degeneracy controls and past-only prefix checks. Do not infer circle membership from future rows.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../docs/data_overview.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [src/wnn/causal_research_features.py](../../../src/wnn/causal_research_features.py)

New code in this PR: src/wnn/causal_research_features.py Shared primitive files have identical contents across related draft PRs, so they should merge once; they are not independent replications. Circle arithmetic is strictFP32, adapted from the source actor commit bc0ead82; cancellation and geometric validity remain untested.
