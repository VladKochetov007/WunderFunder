# Kaggle source-stage operator handoff

Status: preparation only; no provider action or fit has been started. Root retains packet admission and all provider mutations.

## Current runway

Root's fresh read-only Kaggle snapshot at 2026-10-08 21:46:10 UTC reports GPU campaign headroom **24,722.978 s**, TPU campaign headroom **6,936.192 s**, and both reserved totals at **0 s**. Kaggle returned HTTP 429 on the TPUv7 session-status query; its last known state was QUEUED at 21:21 UTC, so current TPU status is unknown. Do not repeat the status request immediately.

The smoke4 receipt is terminal `ERROR`, collected, and its 15,600-second provisional hold is closed. Its boot and nested mount discovery succeeded, and the two T4s were visible. In the last 722 seconds before the worker stop, cgroup-available memory fell from 32,135,933,952 to 1,899,220,992 bytes while combined worker RSS remained about 1.17 GB; no model update was recorded. The packet did not capture `memory.stat` anon/file breakdown, so this is not enough to attribute the decline conclusively to file cache. Repeating smoke4 unchanged is not justified.

Read-only local disk check at 21:47 UTC: `/home` had 30,469,017,600 bytes free and `/tmp` 23,198,011,392 bytes free. Recheck before any artifact collection; the 7.73 GB cache should flow through Kaggle notebook outputs rather than a local download/re-upload.

## Earliest useful action

Run the actor's exact CPU-only cache build and read-only mmap profile on Kaggle, then use that successful kernel's outputs as an input to the admitted two-T4 successor. The actor's pinned membership is fold0, 95/3,148 whole components, 350 fit-train groups, 28 fit-valid groups, and all 468 evaluation groups. Pins are:

- Fold split: `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`
- Portable input registry: `90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c`
- Membership: `dbd9ea6d7b0b752495745ea1ebd4bd858af9305c40f0ada7de7386c5359f3acd`

The CPU kernel must discover the nested `wunder-funder` mount and bind exact `train.parquet` / `valid.parquet` paths. It hashes 34,855,728,139 raw source bytes once, decodes 846 selected complete sequences (about 7.58 GB of feature payload), and writes 7,732,440,000 bytes of cache payload plus manifests/receipts. Root must set measured disk reserves. Its success receipt must include producer manifest and source-verification hashes, full raw-source hash/parse/decode/write telemetry, `/proc/self/io` deltas, cgroup `anon`, `file`, dirty/writeback, process RSS/PSS, available disk, cache manifest SHA, reader-validation SHA and reader-profile SHA.

The CPU-only source-stage candidate is sealed at `cpu_source_stage_candidate_r1_final/` with title `WNN Parsed Cache Source Stage 20261009 V1` and matching ID `vladkochetov07/wnn-parsed-cache-source-stage-20261009-v1`. Its private source asset is `vladkochetov07/wnn-parsed-cache-source-20261009-r1`; the exact upload file inventory, bundle/code/membership hashes, metadata, and kernel source hash are bound by `cpu_source_stage.packet.json` (SHA-256 `0e04b36e04134b085f8f272e172a636d28c376064943c04c23ac27ba9554abbf`). Kernel code SHA-256 is `5720adf09ac9c826f40f626b1d5401bc54d08f405b7c25312e98497cba77ac11`; metadata SHA-256 is `09e66ba72884e19851e2b0050c5898d17f9b43263ef076b440981b8f64d8f1d4`. GPU/TPU are disabled, `machine_shape` is omitted, and `dataset_sources` pins the private source asset and raw dataset. The worker wall is 3,300 seconds within the proposed 3,600-second provider timeout. This remains a candidate pending root review/admission; no upload or kernel push has occurred.

Focused offline checks pass 6/6 under the shared support lock. They cover nested raw-mount discovery and ambiguity rejection, exact private asset and payload hashes, safe source extraction, undeclared archive entries, disk/output guards, boot-marker ordering, and hashes of the actual seven-file source closure from commit `d5daaa0f90b0362877522703c7e17aa78c19cac9`. Ruff check passes. The generated script is 32 KB and compiles. The source stage has not run, so cache manifest, validation receipt, profile, runtime resource measurements and terminal outputs do not exist yet.

The installed Kaggle CLI 2.2.4 accepts `kernel_sources`; the official [Kaggle kernel metadata documentation](https://github.com/Kaggle/kaggle-cli/wiki/Kernel-Metadata) lists it as a list of `owner/kernel-slug` sources. Official [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) says outputs up to 20 GB in `/kaggle/working` can be reused by a future notebook. Therefore the 7.73 GB cache can be retained as the CPU kernel's output and mounted by the GPU successor, avoiding laptop transfer. The exact nested mount path is runtime-dependent and must be discovered, then the full cache hashes/headers must be checked once in the GPU job before either fit worker starts.

## GPU successor gate

Use the frozen WMSE/Huber matched pair only after the source-stage receipt passes independent review and root admits an exact two-T4 packet. Both arms share the same pinned mmap cache and source receipt; they must not rehash raw Parquet independently or reuse profile-trained weights. Their difference is only `--loss wmse` versus `--loss huber`, with 220 updates each. Save the full 468-group prediction arrays, sidecars, checkpoints and all cgroup/device/phase receipts to SSD. Keep the 28 fit-valid groups out of both fits, as the actor's recipe specifies.

No TPU production fit precedes exact real-device model/optimizer/update/save-reload parity. No Vast rental is in scope under the latest user steering. Root alone chooses source asset and kernel IDs, disk/deadline limits, quota hold, admission hash, push command, and whether a fit may begin after the profile.
