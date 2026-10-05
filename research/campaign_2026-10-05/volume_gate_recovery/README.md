# Oct 5 E427 recovery packet

This directory is a fresh, draft-only controller packet. It preserves the E427 trainer, model implementation, exporter, scorer, original score design, resource caps, and the aligned/stale/no-op arms. It has no launch command. The prior Oct 4 controller and queues remain untouched and paused.

`E464_ADMISSION_RECEIPT.json` records the exact resumed service journal evidence, ordered 17 successful attempts, each sealed spec/code identity, all 65 declared artifact proofs (path, size, SHA-256), sequence hash, and paired report hash. `IMMUTABLE_MANIFEST.json` binds the current recovery controller and build files, the ten fresh draft specs, exact attempt IDs, the frozen E427 design, E464 receipt, fit statistics, parent checkpoint/provenance, and existing portable input hashes. `IMMUTABLE_MANIFEST.sha256` is the manifest digest. The fit-statistics, parent, and provenance files were rehashed in this packet; the 29.24 GB training and 5.62 GB validation tables are pinned by the portable manifest but must be streamed and reverified immediately before admission.

`RAM_FEASIBILITY.md` traces the current loader and allocation path against the successful exact-parent fit receipt. It explains why 23 GiB is not supported by the measured peak plus a 1 GiB margin, and why the current four-worker E427 draft has a larger nominal loader queue than the historical parent run.

The three original model-run names remain in the unchanged scorer because the old E427 fits and predictions were never materialized. All ten runner attempt IDs are new under `research/campaign_2026-10-05/volume_gate_recovery/runner`. The manifest build fails if any declared output already exists; root must repeat that check at sealing/launch.

The resource check is deliberately blocked under current limits: 24 GiB worker RAM plus the unchanged 4 GiB support reserve is 30,064,771,072 bytes, 64,771,072 bytes above the 30,000,000,000-byte parent hard limit before current usage. No cache advice or memory snapshot can waive this arithmetic. Keep all original reservations and GPU thresholds. The packet inventories smoke, fit, export, and score; its full GPU reservation is 18,900 seconds (5.25 hours), plus 1,800 CPU seconds. It admits no new stage after 2026-10-05 07:00 UTC and allows any already-admitted worker to drain under its original timeout.

Bounded validation:

```bash
cd research/campaign_2026-10-05/volume_gate_recovery
python -m unittest -v
python -m py_compile controller.py build_drafts.py build_manifest.py test_controller.py
```

Both passed when the packet was written. Root must review the packet, verify all declared inputs and fresh outputs, collect fresh host/GPU admission after exact-path `POSIX_FADV_DONTNEED`, and seal each needed draft in the new state directory before any launch. This packet itself neither seals nor launches a worker.
