# E488 independent grader source review

Reviewed the independent scorer source and its CPU fixtures without running
its actual-attempt verifier, loading evaluation row groups, or scoring model
outputs. Root must still run the sealed `--mode verify-only` path against the
actual completed E488 receipts before any score mode is admitted.

## Reviewed identities

- `score_duration_independent.py`: SHA-256
  `2291f950ec3af3ab285b5fc11f354d699fc7e244bb441f7c9544f7b495327dc0`
- `test_score_duration_independent.py`: SHA-256
  `48e365f2642d68fb73546ed0712bd9c007ae912a77316e4321f7a8695dcfc339`
- E488 fit, probe, design, and amendment identities are pinned in the scorer's
  `EXPECTED_HASHES`; this review is for the current scorer/test hashes above.

## Review findings

The fit-only boundary is placed correctly for verify-only mode. The program
checks immutable sources/specs, attempt/runtime identities, clean exit and
teardown, unchanged worker/parent OOM counters, memory peaks, and exact
artifacts before reading the canonical role manifest. It then independently
derives normalization from the 24 fit-role feature groups, validates the
probe and all 12 trajectories/24 checkpoints, and returns from verify-only
before `_selected_test_rows` is called. The test-role IDs are known from the
role manifest, but no test parquet rows are read on this path.

The reconstruction independently defines the registered H64 two-layer
readout. The GELU/fixed-mix initial state is byte-compared to frozen E474;
learned gates add exactly the expected per-channel parameters. The gate
equations match E484, including the recurrent arm's per-timestep RMS context
normalization and epsilon. Fit normalization is recomputed from fit features,
and checkpoint model buffers must remain identity mean/scale. Checkpoint
names, hashes, keys, tensor shapes through strict loading, finite values,
parameter counts, and state digests are checked.

Fit receipts require exactly 12 seed/arm trajectories, 1,422 continuous
updates each, AdamW step continuity, the three canonical epoch orders, all
registered telemetry samples and row-group identities, and 24 epoch-1/3
snapshots. Probe receipts require all 12 cells, native GELU/fixed-mix parity,
active IEEE FP32 metadata, clipping, changed parameters, finite losses,
state shapes and allocator peaks. The amended timing is recomputed from both
the Python report and controller-charged probe time; the original 900-second
failure is retained while the reviewed 1,800-second allowance is enforced.

After the explicit scoring boundary, the scorer uses the official clipped
weighted-Pearson metric and selected `need_prediction & is_scored` mask for
all 24 snapshots. It retains full raw targets, predictions, masks and
component identities in the export. The numeric-sorted component ordering
and one shared 2,000-draw bootstrap multiplicity matrix are applied to all
snapshots and contrasts. It emits all nine registered global contrasts per
seed (27 total), plus the target-specific counterparts; the duration
interaction is the paired difference of epoch-3 and epoch-1
recurrent-minus-static effects. There is no cross-seed pooling or selection.

Eight synthetic tests passed under `scripts/limited` at 1 CPU / 2 GiB, and
Ruff plus `py_compile` passed under the same limits. Fixtures cover exact
initial-state parity, all four readout equations, paired shuffles, telemetry
locations, projection arithmetic, component-moment invariance, contrast
inventory, causal outputs, and chunk-state carry.

## Disposition

**GO for the actual fit-only `--mode verify-only` gate.** If that gate passes
on the root-reviewed actual receipts, score mode may proceed under its sealed
CPU resource cap. This review does not report a score and does not access
evaluation labels.
