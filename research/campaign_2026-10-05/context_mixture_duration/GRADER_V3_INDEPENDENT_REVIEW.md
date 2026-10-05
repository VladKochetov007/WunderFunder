# E488 independent grader v3 source review

Reviewed the v3 correction to the fit-only evidence gate. This review is
source/fixture-only: I did not run actual evidence verification, read test
parquet rows, or score predictions.

## Exact reviewed identities

- `score_duration_independent_v3.py`: SHA-256
  `5d9a37fe3ce74d8ea7728ccfb4beae1b398a3f4dc9864a7145952356dde2ca21`
- `test_score_duration_independent_v3.py`: SHA-256
  `9ed98340bb27e3d6135144331a7b7705a360e6a958b80d2a1c64804dd1b0eab1`
- The scorer pins `src/wnn/evaluation.py` for its own metric/bootstrap use,
  but the actual sealed fit spec does not declare that grader-only dependency
  as training code.

## Findings

V3 fixes the two concrete pre-score path failures while preserving the prior
v1/v2 sources and receipts. The scorer's repository root is resolved from the
scorer file's `parents[3]`. Its fit and probe spec validators now require the
exact assembled code and input path inventories from the sealed specs, check
their role separation, and retain a separate explicit hash check for the
grader-only evaluation module. The new synthetic fixture loads both actual
sealed JSON spec documents, validates their full inventories, and rejects a
code-to-input role move and a dropped probe input. This is an exact inventory
check, not a permissive omission filter.

The fit-only/held-out boundary and scoring logic are otherwise preserved from
the reviewed v1 implementation. The root-run actual `--mode verify-only`
remains necessary: it must validate the real attempts, source/spec identities,
all fit receipts and checkpoints, and stop before the test-row loader. Only
after that gate passes should explicit score mode be considered.

I ran all ten v3 synthetic tests under `scripts/limited` at 1 CPU / 2 GiB;
they passed. Ruff and `py_compile` passed under the same limits. I also ran
the v3 `--help` command from `/tmp` under those limits. No actual evidence
verification or evaluation rows were accessed.

## Disposition

**GO for the root's actual fit-only `--mode verify-only` gate using this exact
v3 scorer hash.** This does not itself authorize or report a score.
