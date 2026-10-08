# H230 history-swap source V5

V5 preserves the V4 all-data history-target implementation and adds one
provenance-only compatibility rule for the two already frozen strong-teacher
valid exports. Their exact legacy sidecar schema predates the
`inference_precision` field. V5 binds each missing-field classification to the
V3 cache metadata hash, exact valid sidecar hash, prediction/checkpoint hashes,
run, split, epoch, split manifest, and row-group order, then records
`legacy_unspecified`. It does not infer FP16 for those valid exports. Both
corresponding train sidecars explicitly declare FP16 and are verified as such.

Parent prediction exports retain the strict FP32 and features-only checks.
Generic missing/null precision, changed legacy sidecars, and non-allowlisted
sources remain invalid. The V4 source tree and V2 cache attempt are preserved;
V5 does not rewrite the failure or any prediction artifact.

The V5 snapshot contains source, tests, and small sidecar fixtures only. It
contains no data, weights, predictions, or generated cache arrays. The
separate V5 manifest records its V4 parent, the exact changed source files,
and fixture hashes. The cache builder has not been rerun from this snapshot;
root review and a new identified controller admission are required before a
successor attempt.
