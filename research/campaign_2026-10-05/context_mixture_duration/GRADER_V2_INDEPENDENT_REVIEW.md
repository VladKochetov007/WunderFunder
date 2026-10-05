# E488 independent grader v2 source review

Reviewed the versioned scorer path correction and clean-cwd entrypoint. This
is a source/fixture review only; I did not run the actual-attempt verification,
read test-role parquet rows, or score predictions.

## Exact reviewed identities

- `score_duration_independent_v2.py`: SHA-256
  `ce547c80de463522ea85f925629b0caab90607962c6445c9616397cdca838f19`
- `test_score_duration_independent_v2.py`: SHA-256
  `78121af9a69ba1bf166eee9b6172faf2d85393409d5848a3bfca0d25c26d7320`
- The scorer change from v1 is limited to resolving `ROOT` from the scorer
  file's `parents[3]`, the repository root. The added regression asserts the
  real dataset, repository source, and parent actor paths exist and rejects
  the prior erroneous `research/research` location.

## Checks and disposition

I ran all nine v2 synthetic tests under `scripts/limited` at 1 CPU / 2 GiB;
they passed. Ruff and `py_compile` passed under the same limits. I also ran
the scorer's `--help` from `/tmp` through the same hard limits; it exited 0.
The root regression exercises the actual dataset and source paths without
reading row-group contents.

The v1 failure was a deterministic path-resolution defect, not a scientific
or score result. V2 fixes that exact defect while preserving the scorer's
fit-only evidence gate and score boundary. `--mode verify-only` still checks
specs, attempts, runtime identity, OOM/teardown, artifacts, normalization,
probe evidence, all trajectories and all 24 checkpoints before it returns;
test-row loading remains downstream of that return and is used only in
explicit score mode.

**GO for root to run the actual `--mode verify-only` gate using this exact v2
source hash.** A score run remains conditional on that gate passing. No
evaluation rows or prediction results were accessed for this review.
