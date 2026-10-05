# Independent source review: E486 counterfactual scorer

Reviewed scorer SHA-256
`7718e6ec5e833441ceeae2ca803e0f8bbbb9a9880f98317fd285d7cb5512d99c`, tests
`473a0dc38ea7e1466e5415a6c099239d599acbce5810879607bfd5f77d26a44a`, and
addendum `2b658a6201a91d23859f0bbc27fb591e76795a9`. No E486 test scoring or
actual-test-label read was performed in this review. Only source files and
synthetic tests were examined.

The cross-evaluation implementation matches the registered A/B/C/D cells:
it loads each of the 18 frozen E483 checkpoints, switches only the boolean
candidate shape at inference, carries state across 256-row chunks, and resets
to zero per complete sequence. The standalone reset-after scan matches the
frozen model for both shapes. Predictions are checked for finiteness before
clipping; global and target WP use the existing official metric helper and
`need_prediction & is_scored` mask. The numeric-sorted 16-component bootstrap
multiplicities are shared across all four cells and six contrasts within each
seed/momentum pair. The 36-cell and 54-contrast inventories are asserted.

The diagonal safeguard is strong: all 18 selected raw-prediction arrays must
match E483 exactly, followed by target/global point scores, arm bootstrap
draws, and the nine matched-shape D−A point estimates and intervals. Source,
attempt, runtime, OOM, fit-manifest, normalizer, checkpoint, grade-artifact,
and arithmetic-reproduction checks precede E486's parquet test loader. The
root seal is required for scoring, binds one-CPU/no-GPU limits and fresh output
paths, and output creation refuses overwrites. The synthetic suite exercises
both recurrence modes, the six contrast formulas, a one-value diagonal
failure, 36 cells/54 contrasts and export shapes, and the missing-seal gate.

One reporting distinction must be preserved: `validate_upstream()` calls
`load_e483_reference()`, which reads E483's preexisting `selected_exports.npz`
including the previously exported selected raw test targets before returning
the proof-only status. The scorer's `test_rows_loaded: false` flag therefore
means E486 has not yet loaded test rows from the parquet dataset; it does not
mean that no existing test-label bytes were accessed. Do not describe the
proof-only path as label-free. The source note correctly limits its claim to
no E486 test scoring; any result receipt should use explicit fields such as
`e486_parquet_rows_loaded` and `prior_e483_test_exports_read` to avoid
ambiguity.

Bounded validation completed from the scorer directory:

```text
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  ../../../scripts/limited ../../../.venv/bin/python -m unittest -v test_score_counterfactual
Ran 5 tests in 13.588s — OK

../../../scripts/limited ../../../.venv/bin/ruff check --output-format concise \
  score_counterfactual.py test_score_counterfactual.py — All checks passed
../../../scripts/limited ../../../.venv/bin/python -m py_compile \
  score_counterfactual.py test_score_counterfactual.py — exit 0
```

**Disposition: GO for root-sealed E486 inference/scoring**, with the provenance
distinction above stated in the receipt. No new fit is implied; the result
remains a descriptive crossed-inference diagnostic on the reused exposed
panel, not public confirmation or candidate selection.
