# H230 seed9 parent Kaggle V8 transport audit

Audit date: 2026-10-08 UTC. Scope: static packet and focused actual-consumer
checks only. No Kaggle API call, push, fit admission, provider poll, or training
was performed by this audit.

## Finding

V8 fixes the V7 mount-containment failure by streaming verified copies of the
fold manifest and the containment-sensitive teacher-cache files into regular
files under the project root before the unchanged cache validator runs. The
validator still checks the frozen metadata, ancestry and array hashes and
finite values. The mount predicate and the actual cache consumer both pass on
the current packet and data. This is transport evidence, not evidence that a
profile or fit ran.

The earlier V7 failure was triggered because `validate_all_data_distillation_cache`
resolves the teacher split manifest and requires it to remain inside the data
root; a mounted symlink resolves outside. V8 keeps that guard. Its bootstrap
copies the exact declared bytes to regular in-root paths and checks source and
destination size/hash before cache validation. The independently exercised
consumer then validates the real cache arrays and metadata.

## Packet and evidence

The packet is
[`kaggle_parent_v8`](kaggle_parent_v8/). Dataset ID and title agree at
`vladkochetov07/h230-seed9-parent-v8`; the private kernel is
`vladkochetov07/h230-seed9-parent-v8-runner`. The archive has 83 manifest
members, 11 declared external mounts, 211,038 archive bytes and 779,637
expanded bytes. Its bundle SHA256 is
`b279d7dcf6b0326ecc43c2c4a50f255d7b85b7725f69b7940b2bf5be835019c6`.

The deferred-input check passed on the actual materialized
`main_source_train.npy`: expected and observed size 1,696,960,128 bytes, a
regular file under the project root. It rejected a short file, an escaping
regular path and a missing path; an external symlink is recognized as such
and checked by size, with V8 materialization responsible for making the final
consumer path regular and contained.

The actual frozen cache consumer test first reproduced the legacy rejection
when the split manifest/cache entries were external symlinks. It then used
V8's verified materializer for the actual split manifest, metadata and train
and valid prediction arrays; the unchanged consumer passed and returned the
expected full-array hashes. It also rejected overwrite and wrong-hash
materialization. The train prediction size/hash are 1,696,960,128 bytes /
`d706c979f3fae46258ee9684dc754d02a70c35d142fed5ebaf785ae7e1ded165`; valid is
299,680,128 bytes /
`e464d29ca5ec0c3c07e70e1f8b08afb180b5d35b70b4fb3130e6a8f56d0e1a8c`.
The fold manifest is 1,388,846 bytes /
`09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.

The real-consumer test was run under a 1 CPU / 3 GB task cap and passed. An
earlier 1.5 GB attempt was killed with exit 137 while copying the 1.70 GB
array; treat that as an insufficient support cap, not a scientific or source
failure. The current test commands and file hashes are recorded in the
adjacent JSON receipt.

## Next step and limits

Root owns the independent packet review and any fresh quota/admission decision.
After that review, the exact next packet paths are:

- bundle: `kaggle_parent_v8/bundle/bundle.wnn`
- bundle manifest: `kaggle_parent_v8/bundle/bundle-manifest.json`
- dataset metadata: `kaggle_parent_v8/bundle/dataset-metadata.json`
- kernel entry point: `kaggle_parent_v8/kernel/kernel.py`
- kernel metadata: `kaggle_parent_v8/kernel/kernel-metadata.json`

The V8 bundle is **not** a launched or admitted fit in this receipt. A future
operator must collect actual worker phase evidence and verify the V8 mount and
cache validation before interpreting any provider `RUNNING` status as profile
or fit progress. Do not retry the unresolved V7 creation from this audit.
