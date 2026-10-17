# ADR0041: modular training code and permanent structure checks

Status: accepted user direction, 2026-10-17.

The user requests decomposition and reuse of the3738-line recurrent trainer, preserving numerical/efficiency intent and using python-standards plus uv/Ruff. They explicitly authorize a Sol6.1/medium independent critic for this task.

Keep scripts/train_rnn.py as a compatibility CLI. Put reusable implementation in focused src/wnn/training modules, with explicit run-owned configuration/state and semantic phase boundaries. Preserve CLI defaults, numerical expressions, tensor/device movement, recurrent boundaries, update order, ancestry and strict resume. Reuse canonical implementations from research experiments; do not fork another full trainer.

Enforce300lines/module and200lines/function in this package and its entrypoint, plus annotations/docs and no global/nonlocal. Add lint-only CI for main/research-team and PRs, using uv/Ruff; do not require GPU, supplied data or a queue manager for code lint. Keep the compatibility adapter isolated for historical imported settings/custom-name consumers. Extend it deliberately when an existing consumer needs an export.

Record implementation hashes, including optional dependency-metadata gaps in source-only packets. Historical frozen source snapshots remain immutable; strict continuation across a source change remains invalid. No scientific recipe, numerical precision, metric or compute allocation is changed by this refactor.

[Implementation and independent-review receipt](../../research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md) distinguishes static checks from unmeasured runtime/numerical equivalence.
