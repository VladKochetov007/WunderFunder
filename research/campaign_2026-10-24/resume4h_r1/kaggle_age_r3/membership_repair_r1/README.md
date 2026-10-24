# Portable membership schema repair R1

## Status

The local consumer fix passes a metadata-only fixture built from the actual
fold-0 and fold-1 cache sidecars and frozen subset manifests. The test also
confirms that a changed cache membership digest is rejected. Kaggle version 3
still ran the pre-fix overlay; this packet makes no provider upload or retry.
Root must include the changed consumer source in a new, hash-pinned overlay and
admission before another remote profile.

## Diagnosis

The version-3 worker stopped in
[`age_readout_portable_validate.py`](../../../../../src/wnn/training/age_readout_portable_validate.py)
before cache reconstruction, profile microfits, or target reads. Its consumer
loaded the raw v1 fit-subset JSON and looked for
`subset["manifest"]["selected_members_sha256"]`. That JSON stores
`selected_members_sha256` at the top level.

The mismatch came from crossing two representations in the actual producer
path. [`load_component_fit_subset`](../../../../../src/wnn/training/membership.py)
returns an in-memory wrapper whose `manifest` value is the raw JSON object.
[`cache_metadata`](../../../../../src/wnn/training/online_relationship_sources.py)
reads `subset["manifest"]["selected_members_sha256"]` from that wrapper and
writes the value as `subset_members_sha256` in each cache sidecar. The portable
consumer uses `_json()` directly, so it receives the raw object and must read
the field at its top level. The frozen subset and cache bytes agree; no
membership change is needed.

The frozen source cache execution recipe is
[`EXECUTION_RECIPE_R1.json`](../../../../../research/experiment_drafts/online-relationships-20261022/EXECUTION_RECIPE_R1.json),
bound by the portable machine recipe
[`recipe.json`](../../../../../research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json).
The earlier Kaggle failure and source overlay identities are pinned in
[`SOURCE_PINS_R1.json`](SOURCE_PINS_R1.json). The actual recipe and producer
source hashes are included there.

## Repair and evidence

The consumer now reads the top-level `selected_members_sha256`, requires a
64-character lowercase SHA256 value, and compares both fit and evaluation
sidecars against it. The exact equality check remains active. The
`module_hashes` import in
[`age_readout_portable_identity.py`](../../../../../src/wnn/training/age_readout_portable_identity.py)
is now local to the full-cache validator that uses it, allowing the metadata
consumer and fixture to load without importing Torch.

The fixture JSON files are byte-for-byte copies of the source-produced sidecars
and subset manifests. [`test_membership_repair_fixture.py`](test_membership_repair_fixture.py)
exercises the full `fold_metadata()` path for both folds, then changes one
sidecar membership digest and checks that validation rejects it. It opens no
cache arrays, labels, or masks and does not rebuild a cache or fit a model.

[`TEST_RECEIPT_R1.json`](TEST_RECEIPT_R1.json) records the pass, exact source
identities, group counts, source-qualified row-reference hashes, execution
limits and no-Torch/no-payload scope. The current producer, consumer, recipe,
receipt, milestone and fixture hashes are in
[`SOURCE_PINS_R1.json`](SOURCE_PINS_R1.json).

Run the fixture from the repository root with:

```sh
PYTHONPATH=src WNN_MEM_MAX=150M WNN_CPU_QUOTA=100% WNN_THREADS=1 \
  WNN_RUNTIME_MAX=60 scripts/limited python \
  research/campaign_2026-10-24/resume4h_r1/kaggle_age_r3/membership_repair_r1/test_membership_repair_fixture.py
```

Ruff formatting and linting passed for the two changed training modules and the
fixture. The standard-library [training structure check](TRAINING_STRUCTURE_CHECK_R1.json)
also passed across 98 modules with zero errors; its exact output is preserved
in [the log](TRAINING_STRUCTURE_CHECK_R1.log). These checks do not establish
remote import, cache reconstruction, or profile timing.
