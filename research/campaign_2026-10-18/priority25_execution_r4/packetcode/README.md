# Canonical fold-cache repair helper

`restore_canonical_folds.py` derives a new `chain_folds.npz` from the frozen
2026-09-29 fold manifests and the live feature-only `chains.npz` graph. It does
not read labels or dataset rows and never replaces an input. The output cache
is a derived artifact, not a byte-for-byte copy of main's original fold cache.

The helper pins the main graph SHA256
(`39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766`), all
four manifest SHA256 values, and the original fold-cache identity recorded in
those manifests
(`a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f`). Before writing anything, it verifies the live
graph identity and byte size against every manifest; requires each manifest to
contain the full, identical `(split, row_group, seq_ix, component)` inventory;
checks those source-qualified identities and component IDs against the live
graph; verifies evaluation memberships are a disjoint, complete partition of
all validation rows; and rejects any actual graph component split between
canonical folds. It also checks the old cache covers exactly the graph nodes
and records how many of its validation components crossed folds.

The derived NPZ has `split`, `row_group`, `fold`, and `chain`. The campaign
split loader and manifest builder consume `split`, `row_group`, and `fold`;
existing chain-based scoring and probe consumers also require `chain`. The
chain IDs are recomputed from the live graph's undirected connected components.
For train nodes, `fold` inherits the linked validation component's fold, or is
0 when the component has no validation node. Canonical campaign fitting uses
component membership and does not use train fold values.

Run from the code checkout that contains the helper, with the active Beleriand
cache passed explicitly. Both destination paths must be new and must not exist;
keep the current cache and the new cache in separate locations until the
receipt has been inspected:

```sh
python research/campaign_2026-10-18/priority25_execution_r4/packetcode/restore_canonical_folds.py \
  --root /path/to/active-project \
  --canonical-dir /path/to/main-checkout/research/campaign_2026-09-29/artifacts/splits \
  --graph /path/to/active-project/cache/chains.npz \
  --current-folds /path/to/active-project/cache/chain_folds.npz \
  --output /path/to/active-project/research/campaign_2026-10-18/priority25_execution_r4/receipts/chain_folds_derived.npz \
  --receipt /path/to/active-project/research/campaign_2026-10-18/priority25_execution_r4/receipts/fold_restore.json
```

The JSON receipt binds input hashes and graph node/component counts, canonical
fold row counts, old-cache cross-fold component counts, output fields, fold
array hash, output hash, and output path. The caller should retain both new
artifacts and only then point the isolated campaign root at the derived cache.
