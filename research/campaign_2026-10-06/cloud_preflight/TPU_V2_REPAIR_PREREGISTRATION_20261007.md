# E-463 TPU v2 input-mount repair preflight

Prepared 2026-10-07, Europe/Kyiv. E-463 v1 terminated before importing JAX or
touching a TPU because it assumed flat `/kaggle/input/<slug>` mounts and a
pre-extracted source tree. The observed Kaggle layout is nested under
`/kaggle/input/datasets/vladkochetov07/`; the private E-463 source dataset also
contains `source_bundle.zip` plus its manifest rather than a pre-extracted
`source_bundle/` directory.

V2 resolves the source manifest recursively by its frozen SHA-256, checks the
private/offline/resource flags and archive SHA-256, then extracts exactly the
manifest-listed files. It rejects traversal, symlinks, duplicates, unexpected
members, and size/hash mismatch. The raw dataset root uses the observed nested
mount path. V2 writes mount and extraction receipts before attempting TPU
initialization.

The diagnostic is unchanged: require an actual eight-device TPU backend and
HBM headroom, verify the pinned checkpoint/NPZ/sidecar/model hashes, run one
synthetic full-shape update, and test reload parity. It is a systems
preflight—not a scientific fit, score, or competition submission. Packet
`tpu_packet_v2/` is private, TPU-enabled, GPU-disabled and internet-disabled.

The packet preregistration `tpu_prereg_v2.json` caps provider runtime at 600
seconds and the worker at 480 seconds. The campaign TPU cap is 900 seconds; the
pre-push quota poll must show at least 600 seconds of remaining campaign TPU
headroom and no conflicting TPU job. Any later TPU training requires a separate
allocation and real-data/recipe review.
