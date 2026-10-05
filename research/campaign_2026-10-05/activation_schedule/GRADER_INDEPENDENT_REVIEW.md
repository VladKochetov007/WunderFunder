# Independent source review: E476 scorer

**Decision: GO.** The sealed scorer’s target/mask handling, training provenance,
LR-trace checks, and component bootstrap calculations are consistent with the
registered design. I reviewed code and ran only the bounded synthetic scorer
tests; I did not run its dataset scoring path or use a GPU.

The reviewed source hashes match the E476 CPU-grade seal:

- `score_schedule.py`: `2cc00c6e3ed31d6eed14f9dc67d183a179d5070cbd079a32baf82ffba237dbb7`
- `test_score_schedule.py`: `b5ffbc4e56a8c6ed73fe9917785b7c93b95575d24f5191de3465ce20b803dc3a`

The scorer obtains test data through E470’s frozen cohort loader. That loader
computes the feature mean and scale from fit-role rows only, then applies the
normalization and feature clip; target labels are not used to compute those
statistics. It selects official evaluation rows with
`need_prediction & is_scored`. The independent raw-target re-read compares
those masks and clipped values back to the parquet and rejects selected
nonfinite raw labels. Full-sequence predictions are checked for finiteness
before the same mask is applied, and all arms use the same target array, mask,
row order, sequence counts, and component identities.

The model constructor creates the E474 RNN with its identity `mean` and `std`
buffers. E476 initializes each fit from the ordinary E474 seed state and never
copies the external feature normalizer into those buffers; the scorer separately
normalizes inputs through E470 and then strictly loads each checkpoint. I found
no training path that mutates the buffers. The scorer does not add an explicit
post-load assertion that checkpoint `mean` is zero and `std` is one; that would
be useful defense in depth, but the sealed trainer, strict model construction,
and checkpoint hashes bind the expected identity-buffer behavior.

Runner validation is fail-closed: it binds the exact sealed spec, attempt ID,
argv, runtime file set and hashes, successful unit result, exit code, signals,
teardown, worker and parent OOM-counter deltas, run directory, and exact
manifest/trace artifact paths, sizes, and hashes. The scorer also verifies all
sealed code and input hashes, including the full validation parquet digest,
before loading scored rows.

The three LR traces are reconstructed independently from the fixed formulas.
Their values, hashes, arithmetic sums and means are checked against the saved
receipt. Each arm’s trace hash and every sampled telemetry LR are checked. The
warm-cosine and constant-mean traces have equal arithmetic LR sums; the report
correctly avoids equating that with equal optimizer-update area.

All 20 fixed pair effects use the repository’s centered paired component
bootstrap. The two difference-of-differences use common component-count draws
for all four arms and recompute each pooled official WP per draw. Synthetic
tests independently reproduce the interaction draws by expanding sampled
components into repeated rows, check point estimates against official scores,
and cover antisymmetric and zero-effect controls. All 22 contrasts are retained.

Validation run: `test_score_schedule.py` passed all 3 tests and 8 subtests under
the 1-CPU/2-GiB helper. The test suite exercises receipt guards and synthetic
metric/bootstrap calculations without reading prediction checkpoints or
running the actual scoring command.
