# All experiment assets — 1 October 2026

This snapshot preserves **266 run directories, 860 checkpoints, training histories and logs, research records, source history and working-tree patches, and every available C++ submission package** from `beleriand`. It includes failed and aborted runs. The compressed payload is 1,961,928,409 bytes, split into 42 files of at most 45 MiB.

The copy cutoff was **2026-09-30 21:08 UTC / 2026-10-01 00:08 Moscow**. The Transformer fold-1 job was still running; its saved files are a snapshot, not its final results. No training job was stopped. The creative-research iteration is maintained separately in the [research branch](https://github.com/VladKochetov007/WunderFunder/tree/research/creative-iteration-20260930/research/iterations/2026-09-30-creative).

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
| Contents and integrity | [Inventory](inventory.json), [archive manifest](archive_manifest.json), [verification](verification.json) |

V1 uses C's equal-weight four-model recipe (distilled GRU256, GRU128 seeds 3 and 4, chunk-500 GRU128) with two epochs of head fine-tuning. Reported scores and runtimes come from the historical journal; this archive does not independently query the competition leaderboard. Its ZIP SHA256 is `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d`.

## Scope

All available experiment checkpoints, histories, logs and submission packages are included. Bulk prediction/data caches (about 94 GB), organizer dataset bodies (about 33 GB), Python environments and literature PDFs are excluded. Cache and dataset inventories are preserved, small split/fold metadata is included, and [cache restoration guidance](CACHE_RESTORE.md) records how to recover predictions. Original files remain on `beleriand`.

Before/after file size and modification time were checked during copying; the active Transformer files received an additional copy. The final archive passed payload checksum verification, zstd decoding and a tar member listing, including the five empty failed-run directories. These checks establish archive integrity at the cutoff, not completion of jobs that continued afterward.
