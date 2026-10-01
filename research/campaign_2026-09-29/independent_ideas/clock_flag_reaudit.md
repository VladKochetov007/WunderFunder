# Independent quote-cache trainer and builder re-audit

Read-only review on 2026-09-30. This review did not scan Parquet rows or use a GPU.

## Verdict

The `--quote-cache-root` path is wired into both TRAIN and fit-VALID auxiliary
datasets. With the default empty flag, `Sequences` still reads
`ROOT/cache/quote_path_{split}.npy`; the new path does not change the historical
default auxiliary labels. Fold-0 metadata is checked against the clean split,
and both auxiliary variants have the same architecture and output count.

The finished `aux_class_index_f0.json` records 4,495 fit-TRAIN and 528
fit-VALID class-3 groups and explicitly says it read no evaluation groups.
Its code reads only the manifest fit pools' target columns. `build_aux_cache.py`
fits the decoder on the indexed fit-TRAIN class-3 groups, applies that frozen
decoder only to fit-TRAIN/fit-VALID class-3 groups, and creates separate code8
and quote8 arrays. `label_pair` takes the intersection of finite horizon-8
rows, and gives both arms the identical code24 auxiliary second column.
Training gates the auxiliary loss to class 3; zero-filled other groups are
therefore ignored. Evaluation groups are not read by the builder.

## Gate before GPU

At review time the v3 cache metadata and `cache_verification_f0_v3.json` did
not exist, so the paired model runs cannot start yet. Run the completed cache
builder, then `verify_cache.py` to check actual content SHA-256 for all four
arrays, equal finite masks, identical horizon-24 values, and zero evaluation
groups. Its current source *does* hash array bytes. Retain the verification
artifact and require its `cache_metadata_sha256` to equal the metadata SHA used
by both model runs.

The trainer checks each array path exists and that metadata holds a 64-character
SHA, but does **not** hash actual cache bytes. This is a reproducibility gap if
files change after verification. The simplest hard gate is for the trainer to
compare `file_sha256(path)` with `entry["sha256"]` before fitting; alternatively
immutably freeze both arrays after verification and record their actual hashes
in the durable launch record. Its current `array_sha256_expected` field should
not be interpreted as a verified content digest.

`predict_run.py` currently records checkpoint and prediction hashes plus the
split hash, but no auxiliary cache hashes in prediction sidecars. The frozen
screen requires prediction files to carry cache hashes. Copy the checkpoint's
`provenance.quote_cache` into the sidecar before interpreting scores; the
checkpoint SHA already allows indirect tracing, but does not meet that explicit
sidecar requirement.

No leakage or unequal-mask blocker was found in these source contracts.
The class-3 decoder remains TRAIN-target supervised and must not be described
as a label-free physical quote reconstruction. The fold-2 diagnostic used to
choose horizon 8 cannot serve as untouched confirmation.
