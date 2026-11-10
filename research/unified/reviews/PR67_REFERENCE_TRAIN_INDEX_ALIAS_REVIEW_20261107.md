# Independent review of a feature-only train-index compatibility alias

Reviewer: PR67 receiving execution agent, 7 November 2026. This is read-only
inspection of actual Git blobs at the original PR49 inference source commit
`c42baf7b95946994216801b4eb9909422ea7effa`. No original dataset/cache or remote
path was changed by this reviewer.

Disposition: **the proposed explicitly documented receiving alias is compatible
with the exact admitted feature-only export route**. A new receiver-owned
`cache/train_targets.npz` may point to the preserved verified canonical
`cache/train_index.npz`. Preserve the historical target cache unchanged and do
not describe the alias as its original bytes or as a reconstructed label cache.
No source code, split bytes or admission guard needs modification.

The exact control flow is:

1. `scripts/predict_run.py`, with
   `--campaign-full-train-index-feature-only`, calls
   `select_chronological_full_train_index` in
   `src/wnn/training/chronological_teacher.py`.
2. The selector validates the actual fixed checkpoint/run ancestry and then
   opens `cache/train_targets.npz` with `allow_pickle=False`. It reads only
   `targets["row_group"]`; no target, label or mask key is accessed.
3. The selector requires a one-dimensional integer array, checks every official
   training-source row group exactly once, and compares its complete ordered
   array against preserved `cache/train_index.npz["row_group"]`.
4. The predictor sets feature-only mode and uses `FeatureOnlySequences`, whose
   reader calls `SequenceFile.read(..., targets=False, mask=False)` on the
   original Parquet features. The ordinary cached-target dataset path is not
   selected for this export.
5. The exact chronological split implementation separately binds the original
   train index bytes, nanosecond metadata, row-group/sequence map, sidecar and
   source graph. `train_targets.npz` is not among that manifest's source pins.
   All those validations remain active when the compatibility alias is used.

Original Git-blob SHA256 values independently computed during this review:

| File | SHA256 |
|---|---|
| `scripts/predict_run.py` | `981d7557082f8e33e025741dad2b2ab2e86fb20eb4347ab58eca1d53f00c9520` |
| `src/wnn/training/chronological_teacher.py` | `3153be23a9f9e32f4102e3dfde69586d3ea4d4cfeba72570ab3dabcbc53809cf` |
| `src/wnn/chronological_splits.py` | `4d542a619cec8df2e5e524cb564d0dff4b804bcdc6209acb957387422087e8d1` |

The receiver amendment receipt must name the new project, alias path, resolved
target path, original verified index SHA256/size/mtime and identity sidecar, exact
source commit/file hashes, index keys, and successful full ordered-row-group
comparison. Record why the old target cache failed membership comparison and
that its original location remains untouched. Preserve the data-check result and
all bank sidecars/row lists; the importer still requires the complete source
coverage before selecting the declared fitting cohort.

This conclusion is scoped to the explicit feature-only train-index selector.
It does not make this alias suitable for ordinary fitting or any code path that
expects labels or scoring masks. Qualified baseline replay and all subsequent
source, checkpoint, reference and profile gates remain required.
