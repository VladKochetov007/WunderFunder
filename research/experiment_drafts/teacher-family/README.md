# GRU versus Transformer teachers for the same GRU student

Status: implementation handoff; no new fits. Lineage `connection`; `H-DRAFT-TEACHER-FAMILY-20261017` / `E-DRAFT-TEACHER-FAMILY-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. User proposal; historical successful KD does not establish Transformer ancestry.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Construct free-running fold-clean teacher caches, match student seeds, freeze target calibration, add assistant only as separate lineage if justified.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/build_distill_cache.py](../../../scripts/build_distill_cache.py)
- [scripts/train_rnn.py](../../../scripts/train_rnn.py)
- [src/wnn/causal_teacher.py](../../../src/wnn/causal_teacher.py)

New code in this PR: src/wnn/causal_research_features.py, src/wnn/causal_teacher.py Shared primitive files have identical contents across related draft PRs, so they should merge once; they are not independent replications. Circle arithmetic is strictFP32, adapted from the source actor commit bc0ead82; cancellation and geometric validity remain untested.
