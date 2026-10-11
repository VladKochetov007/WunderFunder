# H382 R2 startup repair

R1 (`vladkochetov07/wnn-kaggle-crossfold-h382-20261011-r1`) terminated with a
provider `ERROR` during input setup. Its terminal `status.json`,
`input_identity.json`, and provider log are preserved in the separate
`ops_continuation_20261011/kaggle_h382_delivery_20261011_r1/` receipt. The
error occurred after the source bundle was extracted and ten mounted assets
had matching SHA-256/byte identities, but before raw parquet validation,
`inputs_validated`, any fit, or export. The exact error was an attempt to
symlink the already-extracted fold-0 split manifest over itself.

The split manifest and portable-input registry are deliberately packaged in
the hash-verified source bundle. R2 continues to hash-check both mounted
copies against their asset-manifest entries, then verifies that the already
materialized bundle files have those same expected sizes and digests. It only
symlinks the other mounted assets. Any unexpected pre-existing path,
redirected symlink, or content mismatch remains a hard failure.

R1 omitted `machine_shape` in its submitted metadata and omitted the CLI
`--accelerator` request. Kaggle's pulled R1 metadata later resolved to
`NvidiaTeslaT4`, but R1 failed before device enumeration, so its visible GPU
count is unknown. R2 binds `NvidiaTeslaT4` in metadata and the push command,
then records and requires exactly two visible T4 devices before unpacking
inputs or starting any job. An allocation mismatch is an infrastructure
failure and is not a model result.

R2 has new bundle, dataset, kernel, and run identifiers. Its three preregistered
recipes, fold-0 membership, source snapshot, evaluation exposure, fit/export
bounds, and result contract are unchanged from H382 R1; this is a corrected
delivery attempt, not a new scientific arm. The immutable R1 packet and its
provider artifacts remain untouched.
