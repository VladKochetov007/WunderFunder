# E463 TPU V7 inventory repair candidate

This is an unpushed, source-only successor to the frozen V6 diagnostic packet. V6 reached `host_admission` and failed because its validator expected 24 entries in the `files` array. The pinned private fold-0 asset manifest has 23 payload entries; `assets-manifest.json` is the additional 24th regular file in the mounted dataset.

V7 corrects the inventory gate and a second path mismatch visible in the same frozen manifest: V6 required `model_ep3.pt`, while the manifest names those two run files `model.pt`. V7 stages the exact manifested paths without assigning an epoch label. Its real-data consumer uses fold membership, fixed prediction/propensity caches, and raw parquet; it does not load those two staged checkpoint files. The gate verifies 23 unique, safe manifest payload entries, all required paths, and the exact 24 direct regular-file names in the mount (the 23 payload names plus `assets-manifest.json`). It continues to hash and size-check only assets it stages. It preserves the frozen source, checkpoint, data, and diagnostic recipe identities.

`packet/` and `packet.bundle.zip` are reproducible candidate outputs. The packet manifest binds the exact V6 source and terminal receipt, the reviewed asset manifest, the builder, tests, metadata, and bundle. The test suite builds a tiny temporary mount from manifest filenames; it does not read asset contents, parquet data, or initialize a TPU.

**Provider push is not authorized by this packet.** It establishes only an offline repair candidate. Do not infer TPU device parity or scientific results from these checks.
