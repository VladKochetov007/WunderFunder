# Reusable private fold0 assets

Private dataset `vladkochetov07/wnn-c20261004-fold0-assets` is ready. The explicit
closure from root `continuation_recipes.input_paths(0, parent=None)` contains
23 logical files totaling 7,243,689,021 bytes. Raw train/valid/mask files were
excluded because they are already in the authorized private raw dataset.
Complete content identities are in `assets-manifest.json`; provider filenames
and sizes are in `provider_files.json`. No credentials were included.

Files were staged using hardlinks on the same `/home` disk at
`artifacts/kaggle_assets/wnn-c20261004-fold0-assets`, without duplicating them
into `/tmp` tmpfs. Hashing and upload ran under `wnn-campaign.slice` with one
CPU and 2 GiB RAM. The staging files and original caches must remain immutable.

For each sealed input, pass the mapping
`--external LOGICAL=vladkochetov07/wnn-c20261004-fold0-assets:SHA256.wnn`
and explicitly authorize the dataset with
`--asset-dataset vladkochetov07/wnn-c20261004-fold0-assets`.
The generated kernel binds that exact mapping, attaches only those named
datasets, symlinks the mounted bytes back to the original logical paths, and
the worker verifies all original file hashes before launching.

This closure supports the registered fold0 combo, volume28/current and H256/H384
training recipes. Scientific launches still require the controller's final
code and resource review. It does not include every later experiment's inputs.
