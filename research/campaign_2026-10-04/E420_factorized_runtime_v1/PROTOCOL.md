# E420 factorized runtime probe

This is a separate, not-yet-admitted timing task. Its draft is
`research/campaign_2026-10-04/E420_factorized_runtime_v1.draft.json`; the
runner is `scripts/probe_factorized_runtime.py`. The script fails before reading
model or stream data unless the E420 parity report is bound to the fixed source,
zero-update control, E409 manifest/data, complete source closure, and passing
v2/v3 library hashes. It then verifies both copied libraries against that
report and requires both parity cases plus their bitwise-equality gate to pass.

The measured level is v3, with the verified v2 hash retained as parity
provenance. Three cases run in alternating forward/reverse order for five
trials: exact saved rank-128 `factor_f32`, the same represented control as
effective dense `f32`, and the original W384 parent in `q16x` as separate
deployment-cost context. Timing uses the scorer `DataPoint` loop and native
`EnsemblePredictor.predict`, so it includes per-row callback overhead and causal
event-feature updates. It slices the final head to the first two scored outputs;
the parity run remains the separate all-23-output numerical check. Factor/dense
callback outputs are checked on finite prediction rows after each paired repeat.

The report records the fixed checkpoint and control hashes, stream row groups,
sequence IDs, row counts, selected Parquet compressed/uncompressed input bytes,
resident raw-feature and need-mask bytes, CPU affinity, cycle counter readings,
and wall time. Native memory fields stay separate: `total_streamed_weights`
includes `factor_weights` as a subset, while import-only dense recurrent bytes,
recurrent state, and scratch are their own fields. Python-side parameter import
bytes are separately labeled. No categories are added together and no public
runtime conversion or quality claim is produced.

No timing command has been run. Root admission and a quiet CPU window remain
required before executing the draft.
