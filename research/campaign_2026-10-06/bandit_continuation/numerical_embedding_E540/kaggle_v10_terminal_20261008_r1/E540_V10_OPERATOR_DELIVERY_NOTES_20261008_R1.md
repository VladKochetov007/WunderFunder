# E540 V10 delivery notes

The frozen export packet passed root source and final-byte review. Its V8/V9 checkpoints are already fitted; the V10 worker performs prediction export only.

Kaggle CLI dataset creation from the deep isolated-worktree path failed locally with `OSError [Errno 36] File name too long` while constructing its temporary upload archive. The exact dataset slug was absent in the owner inventory immediately after that failure, so no remote dataset was registered. The first attempt is preserved in `E540_V10_PRIVATE_ASSETS_CREATE_ATTEMPT_20261008_R1.json`.

This Kaggle CLI version flattens the full source path into a temporary upload filename. Subsequent dataset creation uses a short physical staging path, with every staged filename, size, and SHA-256 checked against the frozen packet before upload. The private asset dataset contains nine manifested artifacts plus its manifest; the source bundle dataset is checked separately.

For future Kaggle uploads from this lineage, stage the private asset payload at `/tmp/e540v10a-20261008` and the source bundle at `/tmp/e540v10b-20261008`; do not pass the deep worktree path to `kaggle datasets create`. The first path contains the same nine payload artifacts, `assets-manifest.json`, and Kaggle metadata; its READY v1 download verified the exact nine payload files plus manifest. The second path contains `bundle.wnn`, `bundle-manifest.json`, and Kaggle metadata; its READY v1 download verified the exact two payload files. Kernel files are pushed separately from the frozen packet's `bundle/kernel` directory.

For the kernel push, use the verified two-file staging directory `/tmp/e540v10c-20261008` (only `kernel.py` and `kernel-metadata.json`), after checking both SHA-256 values against the frozen packet. This avoids passing a long isolated-worktree path to the Kaggle CLI while preserving the reviewed payload byte-for-byte.

The long-path failure was followed by an exact owner inventory (`Not found`) before the corrected private dataset create. The source bundle's initial 404 was a read before provider readiness; after `datasets status` returned `ready`, a single download verified version 1. Preserve those failure receipts and check exact owner inventory/readiness before any future retry; an HTTP error alone is not evidence that a dataset or kernel does not exist.
