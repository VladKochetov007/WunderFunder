# E483 independent scorer v3

The frozen v1 scorer/tests remain unchanged. The v1 actual fit-proof gate
exposed an iteration error over `PARENT_PATHS`; v2 fixed that path and its
unmocked sealed-spec identity test passed. The subsequent v2 CPU score attempt
validated all 18 cells and computed the score matrix in memory, then failed
before writing outputs because arm keys were formatted inconsistently at
lookup (`seed...__shape__beta...` versus `seed...__shape_beta...`). Its log and
artifacts are preserved; it is not an official grade result.

V3 is a separately named scorer/tests version. Its `arm_key(seed, shape,
beta)` helper defines the canonical exported key and is used consistently in
cell scoring and selected-export ordering. Within-seed contrast dictionaries
are explicitly mapped to the local `shape_beta` keys expected by
`registered_contrasts`. A synthetic end-to-end test creates dummy checkpoints
for all 18 registered cells, runs `score_all` over 16 synthetic components,
checks all 27 contrasts, invokes `write_outputs`, and validates the complete
18-arm/27-contrast bootstrap export shapes. It patches the scorer's sequence
length to 31 only inside this synthetic test, so it exercises the real
prediction/scoring/export functions without reading the dataset or test rows.

Nine v3 tests pass under the 1 CPU / 2 GiB limited scope, including the
unmocked actual sealed-spec identity check. The source imports and `--help`
command work. No official scoring was run with v3 pending independent source
review.
