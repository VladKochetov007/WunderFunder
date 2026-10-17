# Permitted Chronos or TimesFM teacher transfer

Status: implementation handoff; no new fits. Lineage `teacher/connection`; `H-DRAFT-PRETRAINED-TEACHER-20261017` / `E-DRAFT-PRETRAINED-TEACHER-20261017-R1`. Base main/research-team `18407ecfa3b02d4db70cfb3c34abbacd6653e79d`. Unimplemented proposal; exact pretrained checkpoints/license/input semantics unresolved.

- [English draft PR](PR.md) answers mechanism, compute, validation/significance, ancestry/leakage, decisions, actual code readiness and execution/artifact return. Select permitted public checkpoint, record license/hash/pretraining limits, implement multivariate causal adapter and target mapping; no claimed drop-in model compatibility.
- [Recipe](recipe.json) records strictFP32, candidate/control arms,30-cell protocol (engineering diagnostics excepted), proposed optimizer/scheduler and explicitly missing bindings. Launch-ready is false.
- [Plan entry point](run_plan.py) prints the matrix through the shared [expander](../../../scripts/experiment_draft_plan.py); no fits or queue.
- [Prior evidence/source](../../../research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md) preserves the originating campaign/proposal and its measured/unrun limitations.
- [Unified inventory](../../unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) supplies cross-idea navigation; link future results and returned archive manifests here as they arrive.

## Code and provenance

- [scripts/build_distill_cache.py](../../../scripts/build_distill_cache.py)
- [scripts/train_rnn.py](../../../scripts/train_rnn.py)

New code in this PR: none; reuse linked existing implementation, preserving original source.
