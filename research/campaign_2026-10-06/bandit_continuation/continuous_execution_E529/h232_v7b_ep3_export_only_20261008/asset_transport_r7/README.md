# H232 R7 scoped payload manifest

R6 was accepted and reached the unchanged export worker, which failed before
inference while discovering inputs. The R6 payload dataset added a second
`assets-manifest.json` under `/kaggle/input`; the frozen profile worker's global
`_unique(input_root, "assets-manifest.json")` correctly rejected the ambiguous
mount. The terminal receipt and all downloaded R6 output hashes are preserved
in the `reference/` evidence.

R7 keeps the exact R6 archive, all 18 member hashes, selected fixed-epoch
checkpoint, source archive, and export worker. It changes only the private
payload dataset ID, the payload-manifest filename, the matching bootstrap
lookup, and the canonical kernel ID. The payload dataset manifest is renamed
to `h232-v7b-ep3-payload-manifest-r7.json`; the unchanged worker can therefore
find the one original fold0 `assets-manifest.json` and its hash-addressed
assets. The bootstrap opens its manifest beside the pinned payload file and
validates its exact dataset, filename, size, and SHA-256.

The tests compare the real R6 provider inventories, audit every basename used
by the frozen consumer, and build a three-mount fixture. That fixture calls the
actual unchanged `profile_worker.verify_and_stage_inputs`: it reproduces the
R6 duplicate-manifest rejection and proves the fold0 manifest is selected by
the same recursive lookup after the R7 rename. It does not read Parquet data,
decompress NPZ caches, import Torch, fit, export predictions, or score.

Build the local candidate with:

```sh
python research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008/asset_transport_r7/build_r7_packet.py
```

Run its focused offline checks with:

```sh
PYTHONPATH=src:. python -m pytest -q research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/h232_v7b_ep3_export_only_20261008/asset_transport_r7/test_r7_packet.py
```

The builder performs no provider calls. Root review and fresh quota admission
are still required before creating the R7 private asset version or pushing its
kernel.
