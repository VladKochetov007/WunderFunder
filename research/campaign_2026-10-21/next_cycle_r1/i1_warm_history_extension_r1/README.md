# Conditional warm start for i1 history

This packet prepares a narrow checkpoint continuation for a possible later
matched F0/F1 follow-up. It extends only a standard 211-input GRU parent with
`research_features=none` to the fixed 367-input `i1_history` child, with
`drop_i1=false` and `none` or `all` feature control. The existing canonical
feature-prefix expansion preserves the parent parameters and normalization,
and starts all 156 appended input weights at zero.

The new default-off `--i1-warm-normalization fit-appended` option fits only the
appended mean and scale from the first two batches of the existing fitting
loader, using the same flattened `[::7]` sample and estimator as canonical
normalization. It restores and hashes the inherited 211-slot mean and scale,
and records identities for those buffers and the newly fitted suffix. The
default `inherit` path does not fit or alter normalization.

The source also adds focused synthetic CPU tests for parent-function parity,
sequence/chunk state handling, mask restoration, exact preservation of the
parent normalization prefix, and fail-closed configuration checks. They have
not been run in this preparation worktree; root owns bounded test admission and
review. No tensors from research checkpoints, dataset arrays, model fits, GPU
work, new recipes, or scientific selection were performed here. The mechanism
remains conditional on completion and interpretation of the active i1 and
dynamics screens.
