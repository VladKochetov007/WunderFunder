---
kind: operations
status: complete
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: []
experiment_ids: []
evidence_paths: [MAIN_MAINTENANCE_20261017.json]
---

# Main repository maintenance — October 17

This change implements the user's request to structure the repository and make
Main readable and maintainable. The baseline is
`0edd432271991250f4718f9b68f33340eff46214`. It does not introduce a model
hypothesis, competition fit, prediction export, provider action or submission.

## Navigation and evidence

- Root README, RESEARCH, the documentation index, architecture, development
  guide and repository map now identify the maintained entrypoints.
- Current state is a short dated snapshot. Its previous 199,958-byte contents
  are preserved byte-for-byte in [the same-directory history](../../../specs/current-state-history-20261017.md).
  SHA-256: `37be5efacd617f8e712c99822c0399242949aef10fbeebc9abd20ee90f499902`.
- Ten unified research indexes start with current navigation and preserve their
  dated entries in expandable sections. Past pending/authorization text is
  explicitly historical.
- [Repository audit](../reviews/REPOSITORY_AUDIT_20261017.md),
  [model catalog](../models/CURRENT_MODEL_CATALOG.md),
  [validation protocols](../validation/VALIDATION_PROTOCOLS.md) and
  [storage map](STORAGE_MAP.md) connect existing facts to original evidence.
  [Five compact receipts](../reviews/audit-2026-10-17/README.md) are retained
  with a hash manifest. The leaderboard's 97 means team submissions, not
  competitor count.
- Existing hypothesis, experiment and decision ledgers, frozen campaign source,
  model weights and data are unchanged.

## Code changes

The generic input assembler accumulates feature blocks before the final join.
Ordinary batches avoid repeated concatenations of the growing full-width
tensor. The propensity model still receives its original feature subset and
dtype; sentinel preprocessing remains last. The existing preallocated causal
moment path is unchanged.

Flattening the concatenation graph changed FP32 accumulation order for raw
input gradients in a regression probe. Explicitly differentiable inputs
therefore retain the original staged graph. Tests require exact forward values,
exact input gradients on that path and exact downstream parameter gradients
for ordinary training inputs. Fewer concatenations are established; elapsed
training time and peak GPU memory were not benchmarked.

The legacy Sequences export remains a real subclassable class with constructor
introspection and explicit environment injection. Explicit environment
arguments bypass legacy namespace construction. Serialization preserves normal
importable class/subclass behavior; a transient unregistered facade can be
restored as its canonical dataset class with its captured state. Synthetic
pickle and multiprocessing-spawn checks cover this boundary.

The structure checker reports missing source trees, malformed Python, size
limits and missing annotations without silently passing an empty tree. The
Makefile shares pinned Ruff/static checks with CI and offers a separate bounded
CPU trainer suite. CI adds six stdlib checker tests without installing training
dependencies.

## Verification and limits

The [machine-readable receipt](MAIN_MAINTENANCE_20261017.json) records the exact
commands, environment, source hashes and final outcomes. Checks cover synthetic
CPU feature assembly, causal behavior, real trainer entrypoint/resume/export
integration on tiny synthetic Parquet, compatibility and structure validation.
Two optional real-data tests skip when datasets are absent. The NumPy feature
tests emit warnings from the unchanged research_features.py matrix multiply;
the exact parity assertions still pass.

Final local outcome: **59 passed, two skipped**, plus all Ruff/structure checks,
six stdlib checker tests and CLI help. The six checker tests also occur inside
the 59-test pytest total; they are not six additional distinct tests. The
[full check log](MAIN_MAINTENANCE_20261017.log) retains the output with local
checkout paths normalized.

This suite does not establish CUDA numerical parity, a measured end-to-end
speedup, full-dataset correctness, scorer-platform runtime or model quality.
Original frozen packets retain their source pins and are not silently rebound
to this refactor. The research pause remains in effect.
