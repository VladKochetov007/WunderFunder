# Child all-three profile local checks V1

The [retained proof script](RETAINED_PROOF.py) (SHA-256 `17ae5cbc…1165f`) and its [machine-readable receipt](CHILD_ALLTHREE_PROFILE_LOCAL_CHECKS_V1.json) (SHA-256 `09e3592e…740d7`) document a local, metadata-only check of the child-profile helper closure and canonical CLI. The receipt binds contract `ab68d868…b7c5`, probe `446e679d…be9e4`, support helper `958d973b…f409`, and the 167-file source manifest at commit `68ff682977b822f38351aac0f621146f5f13b440`.

The proof imported all three helper modules from a temporary private-D-style code-only directory, with that directory first on `sys.path`; their observed origins and SHA-256 values are recorded in the receipt. It then imported and hashed all 96 file-backed `wnn` modules (95 `wnn.*` submodules plus the package root), each matching the source manifest. `PYTHONPATH` contained only the pinned `source/src` directory. The real packet template’s trainer argv was parsed through the canonical parser; only the representation-cache and distillation-cache paths were substituted with existing local directories for validation. Training and optimizer entrypoints were not called.

The stored parent terminal acceptance and exact-stop JSON fixtures pass the profile contract validators. The proof did not open the checkpoint, dictionary cache, arrays, or parquet data; its open-audit sentinel recorded zero payload opens, and CUDA remained uninitialized. This is local preparation evidence only: it does not run the GPU profile, verify checkpoint tensor contents, establish model quality, or authorize a launch.

The earlier 17-test synthetic suite result is retained as a pre-inheritance-delta run. After adding the required `inherit_init_preprocessing=True` guard, only the new focused test was run: `test_profile_requires_parent_preprocessing_inheritance` (1 passed). The existing suite was not replayed.

The retained cold proof was run from the maintenance repository with this exact command (the receipt writer is create-only):

```sh
PYTHONPATH=/Users/nb/Documents/wunder/WunderFunder-strong-gru-tweaks-20261019/src /Users/nb/Documents/wunder/WunderFunder-maintenance/.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_profile_local_checks_v1/RETAINED_PROOF.py
```
