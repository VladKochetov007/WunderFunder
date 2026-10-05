# E482 private T4 packet review

The current source packet is assembled locally at `prepared_v4/`. Its archive
has 52 small members; the dataset `datasets/valid.parquet` is hash-bound as an
external mount and is not copied into the archive. Kernel and dataset metadata
mark the draft private. No Kaggle dataset or kernel was created, uploaded,
submitted or run.

The packet retains the six registered seed-271828 recipes exactly: baseline,
sine and cubic candidate maps crossed with AdamW and MuonAdamW. The frozen E481
trainer and E470/E474 helpers are bound by SHA256. Two optimizer-specific
workers are assigned in physical-index order (AdamW to T4 index 0, MuonAdamW
to index 1). Both workers must finish actual CUDA B4×256 updates and native-GRU
plus clipped-optimizer parity checks before any full fit can start. Fit receipts
must confirm the fixed cohort, 474 updates per arm, checkpoint hashes and no
test-row reads.

Admission requires two unique idle T4 UUIDs, indices 0 and 1, at least three
available CPUs, and room for the two 6 GiB worker reservations plus 1 GiB
controller support under the existing 80% host budget. Each worker checks its
UUID and one-device PyTorch visibility, configures and verifies the 2 GiB
allocator ceiling, and is monitored at 6 GiB RSS and 3 GB physical GPU use.
The controller monitors whole-host use and stops at 90%; it takes fresh host
and device inventories after hashing inputs, before each phase, and before
fits. The generated bootstrap pins both archive and manifest hashes, requires
the exact archive member set, rejects paths and links, contains the external
dataset file within its resolved mount root, and preserves the reviewed v7
whole-job watchdog pattern at 3,500 seconds under the 3,600-second kernel
limit. The script refuses new admission at or after 2026-10-05 07:00 UTC.

CPU-only validation:

```bash
.venv/bin/ruff format --check research/campaign_2026-10-05/activation_t4_replication/e482_remote.py research/campaign_2026-10-05/activation_t4_replication/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication/test_e482_packet.py
.venv/bin/ruff check research/campaign_2026-10-05/activation_t4_replication/e482_remote.py research/campaign_2026-10-05/activation_t4_replication/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication/test_e482_packet.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/activation_t4_replication/e482_remote.py research/campaign_2026-10-05/activation_t4_replication/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication/test_e482_packet.py research/campaign_2026-10-05/activation_t4_replication/prepared_v4/bundle/kernel/kernel.py
.venv/bin/python research/campaign_2026-10-05/activation_t4_replication/test_e482_packet.py
```

All four packet tests pass. They exercise the six-cell design, CPU/RAM/GPU
admission failures, probe failure closure and source/input hash binding. They
also build and inspect the actual bundle and generated bootstrap, checking
complete archive membership, omitted-data behavior, manifest pinning, mount
allowlist/path rejection, archive traversal/link/duplicate/missing rejection,
and external symlink containment. No training data rows or GPU were used.

The final local build receipt is `prepared_v4/BUILD_RECEIPT.json`. The builder
formats the generated kernel before calculating its hash, so a fresh build
cannot produce a receipt that changes under formatting. `prepared_v1/` through
`prepared_v3/` are superseded local evidence and must not be used.

| Object | SHA256 |
|---|---|
| Archive | `c9d2b88ef47540cb9ecb13f659552ba5fe1efb351999e76882387d65a2dcc061` |
| Bundle manifest | `db8c0dd1df5b2d9fab1a99c631aabe5af3aa8eb7798a35980570598784b1fb9a` |
| Generated kernel bootstrap | `aed7a7acfc406dd8730bc6bf5d3faada43a79c21d413b0f2b3ee9bee6ca8f3e0` |
| Kernel metadata | `899ae3c236b6632ba49db54d759f001fc5ec87ef4cbfab31fa37fae1fdca856b` |
| E482 worker source | `3398232f34d43ad17aea5aa90675b5d9eaeaacb6e0490815d260dc9966f060b9` |
| Packet builder | `2a449f3050394712b25293e88827e4a42430847392ddd9b7d60832b27810f84e` |
| CPU packet checks | `f6d59dc75423f6a9bff204f50592ea67aa55aa2ff61cf922046e4885ce5821b5` |

Independent source, transport, exact-member and privacy review is the remaining
gate. Before any provider mutation, reviewer/root must confirm these exact
hashes, the resolved Kaggle private dataset slug and mounted `valid.parquet`
identity, dual-T4 allocation and free quota, and current campaign device-hour
budget. The reproducible local packet builder is:

```bash
.venv/bin/python research/campaign_2026-10-05/activation_t4_replication/prepare_e482.py --destination research/campaign_2026-10-05/activation_t4_replication/prepared_review_v5
```

It refuses to overwrite an existing destination and does not call the Kaggle
CLI. Dataset creation or kernel push are outside this source-preparation step;
kernel push starts the remote kernel and requires root admission after review.
