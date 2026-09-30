# All experiment assets — 1 October 2026

The base snapshot plus its [late supplement](delta-late/README.md) preserve **268 run directories, 865 checkpoint paths, training histories and logs, research records, source history and working-tree patches, and every available C++ submission package** from `beleriand`. They include failed and aborted runs. The base snapshot contains 266 runs / 860 checkpoints in 1,961,928,409 compressed bytes (42 parts); the supplement adds two runs and five checkpoint paths in 109,584,984 bytes (three parts). Every part is at most 45 MiB.

The base copy cutoff was **2026-09-30 21:08 UTC / 2026-10-01 00:08 Moscow**. The Transformer fold-1 job was still running then. Its parent, head-only FT and full FT pipeline finished at **21:23:53 UTC**, and a supplementary copy at **21:25:53 UTC / 00:25:53 Moscow** preserves its final checkpoints and [readable results](delta-late/LATEST_RESULTS.md). No training job was stopped. The completed creative-research iteration is maintained separately in the [research branch](https://github.com/VladKochetov007/WunderFunder/tree/research/creative-iteration-20260930/research/iterations/2026-09-30-creative).

## Find the result or experiment

| What | Where |
|---|---|
| Best recorded public submission, V1, **0.6641** | [Download candV1_ft2.zip](submissions/candV1_ft2.zip) |
| All six scored candidates and two engineering packages | [Submission ZIPs](submissions/), [scores and hashes](SOTA_STATE.md) |
| Search all runs, including failures | [TSV index](run_index.tsv), [full JSON index](run_index.json) |
| Research journals, experiment ledgers and individual run reports | [Readable research history](research-history/) |
| Main research code | [quant-probes branch](https://github.com/VladKochetov007/WunderFunder/tree/quant-probes) |
| C++ submission engine | [cpp-engine branch](https://github.com/VladKochetov007/WunderFunder/tree/cpp-engine) |
| Models, histories, logs and exact source snapshots | All `experiments-remote-20261001.tar.zst.part*` files in this directory |
| Restore and verify archive | [RESTORE.md](RESTORE.md), [source reconstruction](SOURCE_RESTORE.md) |
| Latest completed Transformer artifacts | [Supplement and restoration](delta-late/README.md), [measured results](delta-late/LATEST_RESULTS.md) |
| Contents and integrity | [Inventory](inventory.json), [archive manifest](archive_manifest.json), [verification](verification.json) |

V1 uses C's equal-weight four-model recipe (distilled GRU256, GRU128 seeds 3 and 4, chunk-500 GRU128) with two epochs of head fine-tuning. Reported scores and runtimes come from the historical journal; this archive does not independently query the competition leaderboard. Its ZIP SHA256 is `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d`.

## Scope

All available experiment checkpoints, histories, logs and submission packages are included. Bulk prediction/data caches (about 94 GB), organizer dataset bodies (about 33 GB), Python environments and literature PDFs are excluded. Cache and dataset inventories are preserved, small split/fold metadata is included, and [cache restoration guidance](CACHE_RESTORE.md) records how to recover predictions. Original files remain on `beleriand`.

Before/after file size and modification time were checked during copying; the active Transformer files received an additional copy. Both archives passed payload checksum verification, zstd decoding and tar member listing, including the five empty failed-run directories in the base. The supplement separately records the completed Transformer pipeline. The base archive remains immutable; consult the supplement before treating a base snapshot's live-job status as current.
