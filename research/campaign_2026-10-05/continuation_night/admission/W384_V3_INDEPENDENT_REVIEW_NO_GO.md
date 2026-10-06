# W384 transport v3 independent packet review

Review date: 2026-10-06. This verdict is bound only to the immutable private
packet identity below. No provider action was performed.

## Verdict: NO-GO

The exact packet fixes the short-fixture contract mismatch and preserves the
production 20,000-row contract, and its private transport setup is suitable in
the reviewed respects. However, the completion artifact manifest omits the
fresh W384 checkpoint and run provenance because its run-directory glob still
uses the previous `c20261005night` prefix. Rebuild and re-review before any
provider action.

## Packet identity and independent checks

| File | SHA-256 |
|---|---|
| `kernel.py` | `10f832310f5242d0b7f2e999631594418cdabf1d4cb153e3b2d460a304163814` |
| `kernel-metadata.json` | `14346f36b7ea594baf0f98b06d761b0807fffe6b005c2474143d9eccc9426137` |
| `review_packet.json` | `af0831693d2664bd602c9c666d49e641216c72a4ac478928af4e0b165a56370a` |
| Bound bundle manifest | `4518e583e9eab4bcaa5c5eb499958e9329748191006e68cd89a101d12cee1958` |
| Bound bundle payload | `498c0a5d1105a9fbbe34d02fffb7d4a6c12cd5d2f6b10f8ef647b7bac03bb90a` |

Decoded embedded source bytes matched the packet's five source hashes:

- `payload.py`: `f256c64d0854cff451c6c254e580e4f2ce315250911e84288605f2b45784accc`
- `bounded_loader.py`: `e50c47eeabd8c23cc9bcef0f197891a4c06da5acbc48be2f22b0726f1c01dac2`
- `derive_source.py`: `be47273d0e46d4df06caa5ea511173979967e575f8c18c87dc78076042138f0c`
- `phase.py`: `99d755da36650aaf84c14c7603343d51c55bd379f7e013b1240c577c17cfa9e4`
- `host_guard.py`: `8ca8a97186b808c47289f3c379c923f63ca3fbe0e93bb6ea91ba6f43536ebdfa`

The loader now resolves the active `wnn.data.SEQUENCE_LENGTH`, so the eight-row
integration fixture follows the same contract as the reference loader. Before
runtime staging, `derive_source.stage()` verifies the manifest-bound data
source and statically requires exactly one top-level `SEQUENCE_LENGTH = 20000`.
Row-group metadata, decoded overrun/final length, and sequence IDs are checked;
features, targets, and masks use the same bounded decoded batches. The derived
trainer transformation remains loader-only, and the exact derived trainer
hash matches the previous recovery source.

The supplied receipts record 36/36 manifest-source CPU preflight tests and
16/16 transport tests, both with zero failures/errors, plus clean Ruff. This
review did not rerun those unchanged suites. The packet metadata is private,
sets internet and TPU off, and contains no Kaggle API push or launch step. The
only provider entry point is this W384 packet; the embedded call fixes width
384, worker concurrency is one, and no score computation is present.

## Material blocker

`phase.py` sets `fit_name` to
`c20261006night_w384_adamw_s3_f1_serialv3_fit` and confirms
`artifacts/runs/{fit_name}/model_ep3.pt` exists. But it builds the run-artifact
manifest with `artifacts/runs/c20261005night_w{width}_*/*`, then separately
adds only `required[1:]` (the validation prediction array and sidecar). The
fresh `model_ep3.pt`, provenance, history, and training state therefore do
not enter `artifact_manifest.json`. Change the collector to bind the exact v3
`fit_name` outputs, including the model and provenance, and rebuild the packet
and source receipt. The stale source hashes remain a NO-GO and cannot be used
after that change.
