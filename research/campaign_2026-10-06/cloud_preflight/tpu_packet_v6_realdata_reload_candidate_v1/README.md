# E463 TPU real-data reload probe V6

This is a source-only successor to the completed E463 V5 TPU diagnostic. It
retains the V5 eight-device synthetic reduction, finite-value, and HBM checks,
then adds exactly one real-data B16/C2000 update using the same independently
trained `c20261003_combo_s3_f0` epoch-3 checkpoint. It verifies fold/source
membership, serializes and reloads updated student, EMA, and AdamW state, and
checks causal prefix invariance plus zero-state reset behavior.

The update reads only 16 deterministic, source-qualified `valid` fit groups
from the frozen fold-0 set (10,535 train, 1,405 fit-valid, 468 evaluation).
Those validation groups are exposed training examples, so this is an execution
and checkpoint-compatibility gate, not a held-out evaluation. The worker emits
no predictions, metric, score, or public forecast. No training is done locally.

The equation, optimizer, source, and membership mapping is recorded in
[V6_RECIPE_AUDIT.md](V6_RECIPE_AUDIT.md).

The parent verifies the pinned E463 source manifest and files, the fold-assets
manifest and each consumed file, the two raw parquet file sizes and SHA-256
identities, and the embedded epoch-3 provenance. It stages read-only inputs by
symlink; the 35 GB raw parquet pair is hashed once before staging. The runtime
source imports the exact E465 `distributed_combo.py`, `e465_host_contract.py`,
and `onecycle.py` bytes whose hashes appear in `packet_manifest.json`.

The worker timeout remains 1,500 seconds inside a 1,650-second parent bound.
The previous V5 run completed its synthetic TPU checks in 68.4 seconds, but
that does not estimate the new raw-data hashing, real-data batch build, or
checkpoint reload. Root admission and an actual current TPU quota check are
still required before any provider action. This packet is not authorized for
push.

Build with `reference/build_v6_probe.py`. The generated provider payload is
`packet/probe.py`, bound with `packet/kernel-metadata.json` and
`packet.bundle.zip`. Run the focused, CPU-only contract suite with
`pytest -q tests/test_v6_packet_contract.py`; it does not open parquet/NPZ data,
import JAX, load model weights, or contact Kaggle.
