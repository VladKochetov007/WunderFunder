# R2 V2 terminal capture

This folder preserves the completed, already-authorized CPU-only R2 V2 invocation. `REMOTE_CAPTURE_TRANSPORT.json` is the exact local SSH stdout envelope; its embedded base64 payloads were checked against the declared byte lengths and SHA-256 digests before exact decoded copies were written here. `REMOTE_CAPTURE_TRANSPORT.stderr.txt` is the separate SSH stderr. No arrays or large artifacts were copied.

The exact unit `computer-vision-homework-eda19-r2-v2.service` ran from 2026-10-18 22:34:56 UTC to 22:37:36 UTC, with `Result=success` and `ExecMainStatus=0`. After all raw captures were durably written and verified, a guarded stop targeted only that exited unit; [stop receipt](REMOTE_STOP_RECEIPT.json) records stop return code 0 and the subsequent inactive/dead state. The launch limits were CPU quota 50%, MemoryHigh 1,400,000,000 B, MemoryMax 1,500,000,000 B, swap 0, and 600 s runtime. The unit property snapshot confirms the memory and swap limits. `DONE.json` reports 159.873 s elapsed, 79.903 s process CPU, 745,152 KiB max RSS, CUDA not initialized, and six views. Systemd separately reported MemoryPeak 312,877,056 B; both instrument readings are preserved because they differ.

The exact remote terminal files are [START](REMOTE_START.json), [source manifest](REMOTE_SOURCE_MANIFEST.json), [DONE](REMOTE_DONE.json), [RESULT](REMOTE_RESULT.json), and the complete [incremental progress JSONL](REMOTE_INCREMENTAL_PROGRESS.jsonl). [Root admission](REMOTE_ROOT_ADMISSION.json), [root launch](REMOTE_ROOT_LAUNCH.json), the bounded [log tail](REMOTE_ANALYSIS_LOG_TAIL.txt), [unit status and journal](REMOTE_UNIT_STATUS.txt), [unit properties](REMOTE_UNIT_SHOW.txt), [campaign/parent slice snapshots](REMOTE_CAMPAIGN_SLICE_SHOW.txt), and [cgroup snapshot](REMOTE_CGROUP_SNAPSHOT.json) preserve resource and event evidence. The log tail was empty. The campaign-slice events at capture were `high=757`, `max=561`, `oom=0`, `oom_kill=0`, and `oom_group_kill=0`; these are aggregate slice counters. The worker's live cgroup `memory.events` was all zero at the last live check before exit; its transient cgroup had been removed by terminal capture.

All six selected outer metrics use 35,467 rows and the fixed `global_wp_clip2_v1` metric. Columns below are view, selected alpha, `t0`, `t1`, and WP:

| View | Alpha | t0 | t1 | WP |
| --- | ---: | ---: | ---: | ---: |
| `raw` | 100 | 0.6804723966 | 0.6968340787 | 0.6886532377 |
| `i0_only` | 100 | 0.6812344347 | 0.7151520589 | 0.6981932468 |
| `raw_lags_1_16_128` | 10000 | 0.6809888776 | 0.6961797796 | 0.6885843286 |
| `raw_signed_delta_ema_hl32` | 10000 | 0.6589827003 | 0.6742348797 | 0.6666087900 |
| `raw_absolute_innovation_ema_hl32` | 10000 | 0.6628312237 | 0.7015282833 | 0.6821797535 |
| `raw_both_ema_parts_hl32` | 10000 | 0.6688806085 | 0.7084460949 | 0.6886633517 |

`RESULT.json` identifies 54 inner model fits, component-group folds, 48 source pins, and the sample manifest SHA-256 `0b03a4e2a6aab4eca36bdbb721d30e94b675c3425c0e0a7deca5ae562c2bbaa5`. The source manifest records canonical snapshot `source-a266b036`, the `global_wp_clip2_v1` implementation SHA-256 `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa`, and states that the R1 loader verifies each selected NPY size and hash before returning read-only memmaps. The completed result explicitly uses float64 normal equations and is exploratory only; its outer panel is the same exposed R1 panel, so this is descriptive screening evidence, not strict-FP32 campaign training, confirmation, or a fresh evaluation.

Local integrity references: raw transport SHA-256 `d18b7d8d29ca9b434dff7356e96d81929efebecf66dd32bb35f667b2fef9541e`; RESULT SHA-256 `ea646da1fa8fd11d89ed6251932dba242e64163c0337dca6bdb8bf377839a967`; source-manifest SHA-256 `ca01c994567fac44e2069d3230ba74f6253470100a3e534d4b4387b8480bddcb`.
