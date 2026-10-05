# E482 worker CLI repair v2

This candidate repairs the v1 bootstrap dispatch error. The v1 provider job
stopped before CUDA setup: both worker logs report the required `--output`
argument missing, while the outer orchestrator's `parse_known_args()` had
already consumed and discarded that option before calling `worker_main()`.
The preserved incident evidence is in the original v1 directory at
`activation_t4_replication/remote_v1/e482-results/`; its provider receipt shows
both T4s idle, approximately 24 GiB host memory available, and all cgroup OOM
counters zero. This was a command forwarding failure, not a model or resource
failure.

The v2 dispatcher detects `--worker` in the original argument vector and passes
all remaining tokens directly to the strict worker parser. A subprocess test
invokes the actual script for both `probe` and `fit`, with both optimizers and
`--parse-only`; each child must echo the exact output path without loading data
or initializing CUDA. The v2 test suite also rebuilds and inspects the sealed
bundle, manifest, private metadata, extraction guards, and external mount
containment.

The packet remains the same E482 six-recipe scientific design and keeps all
prior source, worker, GPU, RAM, host admission, watchdog, cache, mount, archive,
and teardown guards unchanged. Only the dispatcher, v2 source binding, private
dataset identity, and kernel identity differ. The dataset remains external and
is not copied into the archive. Kernel `id` and `title` are both exactly
`vladkochetov07/wnn-e482-t4-sixfit-20261005-v2-runner` and
`wnn-e482-t4-sixfit-20261005-v2-runner`, respectively.

The local candidate is `prepared_v1/`; it has 52 archive members and no raw
dataset. The build receipt reports `local_source_packet_built_not_uploaded_not_launched`.
No dataset was created, no kernel was uploaded or submitted, and no probe or fit
was run. This packet is awaiting independent source and transport review.

CPU validation:

```bash
.venv/bin/ruff format --check research/campaign_2026-10-05/activation_t4_replication_v2/e482_remote.py research/campaign_2026-10-05/activation_t4_replication_v2/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication_v2/test_e482_packet.py
.venv/bin/ruff check research/campaign_2026-10-05/activation_t4_replication_v2/e482_remote.py research/campaign_2026-10-05/activation_t4_replication_v2/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication_v2/test_e482_packet.py
.venv/bin/python -m py_compile research/campaign_2026-10-05/activation_t4_replication_v2/e482_remote.py research/campaign_2026-10-05/activation_t4_replication_v2/prepare_e482.py research/campaign_2026-10-05/activation_t4_replication_v2/test_e482_packet.py research/campaign_2026-10-05/activation_t4_replication_v2/prepared_v1/bundle/kernel/kernel.py
.venv/bin/python research/campaign_2026-10-05/activation_t4_replication_v2/test_e482_packet.py
```

All five tests pass, including four real child-process CLI cases. Ruff and
`py_compile` pass for the sources; the generated kernel formats and compiles.

| Object | SHA256 |
|---|---|
| Archive | `6cf83debabd43ef46365d224492c31f3c33aa579f07190ae387dc5384ccd4245` |
| Bundle manifest | `64796bd729527c70a0be5402979be1bd5e5d0c525207cc57fb2fb0a3c38aa473` |
| Generated kernel bootstrap | `2d630c8c8d6d43d23c0053bc4c3f4d6c1a22ce293510754ac14312529fa2cf0c` |
| Kernel metadata | `8935ac6dfe870891c8c6d2fc4308e0dac2bdf17c697528d5ed3ffe7e39894eb0` |
| Dataset metadata | `53366f288ec12610679b518784b7c6a8ffd36a2b8f3e779ffaa83eb573f2d52a` |
| E482 worker source | `ad8bc19fd46a1394b7f9346eeb7451e14f7e4a6fc6bb643fdb6584d9566e4982` |
| Packet builder | `9acb6bd62b43f6cae66a09847ed9d2ff8dc37e6d3f12e009a23ede4ccd6fed54` |
| CPU packet checks | `94d1049c3bdf3d2c9bd23095a1e13c5352c1b6fcf441b01334267d4f8f38bc2c` |

The builder refuses to overwrite an existing destination and makes no provider
calls. Independent review must confirm these hashes and the exact mounted input
identity before any provider mutation.
