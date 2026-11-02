# Nested OOF blend shrinkage

`analyze.py` fits fixed shared weights on inner OOF predictions, shrinks them
by a predeclared amount toward the existing blend, then scores the frozen
candidate and baseline on a separate outer component panel using the pooled
`global_wp_clip2_v1` implementation. This is analysis, not model training,
serving validation or a claim that OOF data already exist. The solver has three
inner starts and ranks endpoints on the same inner panel; record that selection
exposure. Its solution is a bounded local optimum, not a global guarantee.

Run on a provisioned Linux receiver with the project environment, NumPy and
SciPy. Do not run on the local machine. From the checkout:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Obtain this fixed digest from the independently reviewed manifest receipt.
export FROZEN_MANIFEST_SHA256='<full 64-character SHA-256>'
python research/experiment_drafts/gold-frontier-hypotheses-20261102/nested_oof/analyze.py \
  --manifest /absolute/path/to/frozen-manifest.json \
  --manifest-sha256 "$FROZEN_MANIFEST_SHA256" \
  --output /absolute/path/to/new-result.json
sha256sum /absolute/path/to/new-result.json
```

No outputs are overwritten. Scripts need hash-qualified source datasets,
produced predictions and full lineage before analysis; the placeholder
hash and paths are not launch-ready. New training/export is needed if they are
unavailable. Obtain datasets/predictions by the competition's allowed data path.
The tool does not create them and does not infer their provenance.

## Manifest contract

All artifact paths below are **relative to the manifest directory**, resolve
inside it, and carry full lowercase SHA-256 hashes. Hashing verifies byte
identity, not receipt semantics. Make immutable, read-only copies to prevent
concurrent changes. This example describes the schema; names and hashes are
placeholders and it is not an actual receipt:

```json
{
  "schema": "wnn.nested-oof-shrinkage.v1",
  "members": ["A", "B"],
  "baseline_weights": [0.5, 0.5],
  "shrink_to_baseline": 0.5,
  "ridge": 0.0001,
  "source_datasets": [{"path": "sources/train.parquet", "sha256": "REPLACE"}],
  "operator_attestation": {
    "operator": "NAME",
    "complete_ancestry": true,
    "component_membership_verified": true,
    "row_alignment_and_masks_verified": true,
    "receipts_reviewed": true
  },
  "lineage_nodes": [
    {"id": "a_inner_1", "kind": "model", "member": "A", "fit_components": ["c2"], "parents": ["norm_inner_1"], "receipt": {"path": "receipts/a1.json", "sha256": "REPLACE"}},
    {"id": "norm_inner_1", "kind": "normalizer", "fit_components": ["c2"], "parents": [], "receipt": {"path": "receipts/n1.json", "sha256": "REPLACE"}}
  ],
  "inner": [
    {"components": ["c1"], "producers": ["a_inner_1", "b_inner_1"], "data": {"path": "predictions/inner1.npz", "sha256": "REPLACE"}},
    {"components": ["c2"], "producers": ["a_inner_2", "b_inner_2"], "data": {"path": "predictions/inner2.npz", "sha256": "REPLACE"}}
  ],
  "outer": {"components": ["c3"], "producers": ["a_outer", "b_outer"], "data": {"path": "predictions/outer.npz", "sha256": "REPLACE"}}
}
```

Complete the example with nodes and receipts for every referenced ID; the
shown fragment deliberately fails validation. Add every model/checkpoint,
teacher, initialiser, selector, transform and normalizer recursively as a node
with all fitting-component IDs and parents. Root producer node `member` must
match its column in `members`, and `kind` must be `model` or `checkpoint`.
Each inner producer and every ancestor must exclude that inner part's components
and the entire outer set. Every outer producer and ancestor excludes the entire
outer set. Inner component sets must be disjoint from one another and outer.
The selected fitting data can be restricted, but all missing predictions and
related compute costs must be recorded. Do not use PR54 all-label weights for
any part whose components they saw.

Each NPZ contains exactly four arrays with common row order: `component` (1D
fixed-width unicode or bytes component ID), `target` (N×2), `prediction`
(N×M×2 in member order), and `mask` (N boolean official score mask). Each part's
observed component IDs must equal its declared components. Supply complete
sequence rows, true official warmup/is_scored masks and source row ordering;
never substitute output indices for component identity. Up to five million rows
per part and inner total, eight members, 1.5 GB uncompressed per NPZ, and a
4 MB manifest are admitted. Full pool WP, target WPs and selected counts are
returned. The output records the frozen manifest SHA. Fresh evaluations are not
created by renaming exposed components.

**Attestation boundary:** receipt files are hash checked but not parsed as a
universal receipt format; this tool cannot independently prove that the declared
DAG is complete, that its source membership is truthful, or that NPZ rows and
masks match the original dataset. A named operator must inspect those original
receipts and preprocessing/normalizer/checkpoint lineages and attest each field.
Unknown or incomplete lineage must remain blocked. These claims are therefore
operator-attested, not automatically verified. Never edit predictions or labels
to make a manifest pass; freeze the full selection procedure before accessing
outer scores. An exact submission ZIP still needs separate causal and one-vCPU
quality, parity, size and runtime checks.

No tests, model inference or fitting were run locally for this implementation.
