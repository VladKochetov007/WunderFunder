# F3 teacher CP2 export packet R2

R2 is a quota-safe successor to the frozen R1 packet. It keeps the same verified F3 seed-3 H384 epoch-2 checkpoint, provenance, split, prediction scopes and 18,000-second export worker cap. The generated private kernel bootstrap enforces a 21,600-second outer wall limit, emits a matching timeout diagnostic, and leaves a 3,600-second startup/finalization margin. R1 files and hashes remain unchanged; R2 uses its own bundle and kernel slug so it cannot silently replace the prior packet.

The worker serially exports causal FP32 predictions for 10,539 fit-train, 1,368 fit-valid and 1,873 full-valid sequences. Their raw array data total 2,204,800,000 bytes; the three collected NPY files total 2,204,800,384 bytes including headers. It performs no fit, label-based score, grade or selection. Prediction runtime has not been measured; the 18,000-second worker bound preserves the three existing 6,000-second per-export caps. The 8 GB host reservation, 12 GiB CUDA allocator cap, 13 GiB observed stop and 6 GB free-work-disk gate are unchanged from R1.

The bundle slug is `wnn-f3-t3-cp2-exports-20261015-r2`, and the kernel id is `vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r2-runner`. The private CP2/provenance asset dataset identity stays `vladkochetov07/wnn-f3-t3-cp2-assets-20261015-r1`, whose local content-addressed stage is verified by the builder before packaging. R2 was pushed as private kernel version 1 and reached provider `COMPLETE`; this export-only run makes no fit or teacher-quality claim.

## Offline build and checks

Run the local-only builder from the repository checkout after root review:

```sh
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-15/f3_teacher_export_r2/prepare_export_packet_r2.py \
  --root /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015 \
  --source-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --rebuild-local
```

The builder verifies the exact CP2 and provenance bytes in the existing private asset stage, constructs a new R2 job and bundle, lowers the generated kernel bootstrap deadline to 21,600 seconds, then hashes the resulting bootstrap, metadata, source binding and archive. The offline preflight fails if the generated kernel id, outer cap, asset mounts, bootstrap size, archive hash or declared output list drifts.

## Conditional provider command sequence

Only after an independent critic gives GO, a fresh typed GPU quota has at least 21,600 seconds remaining, the owner inventory has no conflicting job, and root admits this exact hash set may the assigned delivery operator run the following private operations under root's authority. The earlier quota snapshot at `2026-10-15T11:50:33Z` is historical and does not authorize launch.

```sh
kaggle datasets create -p /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/kaggle_assets/wnn-f3-t3-cp2-assets-20261015-r1
kaggle datasets create -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/bundle_wnn-f3-t3-cp2-exports-20261015-r2
kaggle kernels push -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/bundle_wnn-f3-t3-cp2-exports-20261015-r2/kernel
```

Root must verify both datasets are private and READY at the exact versions mounted by the kernel, and write `ROOT_ADMISSION_R2.json`, `ASSET_DATASET_READY_R2.json`, `BUNDLE_DATASET_READY_R2.json`, and `KERNEL_PUSH_R2.json`. The admission binds the 21,600-second outer cap and 18,000-second worker cap; the push receipt binds the positive kernel version, archive and metadata hashes, and private status. No automatic retry is authorized.

## Terminal collection

The first R2 collector invocation reached terminal `COMPLETE` but stopped before creating a collection directory or downloading any result body: its exact allowlist omitted the standard runner's `results/heartbeat.json`. The frozen failure is in [COLLECTOR_INVENTORY_FAILURE_R1.json](COLLECTOR_INVENTORY_FAILURE_R1.json). A read-only response-header inventory is in [REMOTE_OUTPUT_INVENTORY_R1.json](REMOTE_OUTPUT_INVENTORY_R1.json): all 15 result files total 2,205,066,956 bytes, the session log is 34,396 bytes, and the total with the 4 MB manifest reserve is 2,209,101,352 bytes, below the 2.5 GB cap. The sole additional path is the 502-byte heartbeat written by the exact runner source pinned in the deployed archive.

R2's collector remains frozen. The narrow [R3 collector](collect_terminal_r3.py) verifies the archived GPU runner hash, admits only that exact heartbeat path with a 16 KiB declared and downloaded size limit, and validates its small JSON against the runner's resource-only schema. It preserves the R2 source/CP2/provenance bindings, six prediction files and sidecars, 2.5 GB cap, streaming length checks, and finite FP32 array scans. It writes to `collected/attempt_002` and records `TERMINAL_COLLECTION_R3.json`; it does not rerun or repush the kernel.

The admitted R3 collection used this command against the already-complete R2 kernel version:

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=3000000000 WNN_RUNTIME_MAX=3600s \
scripts/limited /home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-15/f3_teacher_export_r2/collect_terminal_r3.py \
  --repo-root /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015
```

The collector requires the exact R2 admission and push receipts, inventories the complete `results/` allowlist via GET headers before downloading, streams at most 2.5 GB, and writes on the physical filesystem with a 4 GB free-space floor. The R3 invocation exited successfully and [the terminal receipt](TERMINAL_COLLECTION_R3.json), SHA-256 `1dd10dd2bbe4b3a7059eaf620089dafef2bf5a628d08bc66043a02984f133054`, records provider `COMPLETE`, all expected outputs, matching worker manifest and prediction sidecar hashes, the worker pipeline's CP2/provenance/split identity, FP32 shapes and a bounded mmap scan confirming all 551,200,000 prediction values are finite. Total NPY bytes include 384 bytes of headers. The full-valid prediction is available at `collected/attempt_002/results/export_f3_teacher_cp2_triplet_20261015_r2/artifacts/cache/pred_clean_t3_f3_muonall_h384_e3_s3_r5_full_valid_valid.npy` with sidecar at the same path ending in `.json`; its array shape is `[1873, 20000, 2]`, its NPY size is 299,680,128 bytes and its SHA-256 is `ec9a993a4e91d5ea7ed08c2884bffe09d12b4741e60539680929a404abc2e5c5`. Small collected JSON receipts, sidecars and logs are versioned; the three NPY prediction payloads remain in physical storage and are not committed. Root owns official F3 grading, which is pending; this packet itself makes no teacher-quality claim.

The local collector ran with a 3 GB memory cap and 1 CPU; its transient scope was absent after exit, with swap still at zero. The post-command slice snapshot was 7,028,137,984 bytes current and 16,144,187,392 bytes cumulative `MemoryPeak`. That high-water value is shared slice telemetry, not attributable to this collector; no command-specific peak was captured.

## Provider preflight receipts

- [Initial provider preflight](PROVIDER_PREFLIGHT_R2.json) preserves an incomplete check that inferred dataset IDs only from object-valued cache mounts; it omitted `wunder-funder`, whose raw parquet mounts are represented as strings. No provider action followed that receipt.
- [Corrected provider preflight](PROVIDER_PREFLIGHT_R2_CORRECTION_R1.json) derives the full dataset-source set from frozen kernel metadata and verifies all three existing mounts are private, version 1 and READY. At 2026-10-15 12:39:31 UTC, typed GPU quota had 26,627.389 seconds remaining against the 21,600-second R2 gate, all 176 owner kernels were inventoried, the 16 kernels run in the prior four days were terminal, and both R2 dataset targets and the kernel slug were absent. It also rechecked the local content-addressed CP2/provenance files, private dataset metadata, 21,600/18,000-second outer/worker caps, and bundle hash. This finding establishes operational eligibility under the root authority; no provider mutation had occurred when the receipt was written.

## Navigation

- [Source and ancestry binding](SOURCE_BINDING_R2.json) pins CP2, provenance, source-qualified split, frozen predictor and generated R2 kernel artifacts.
- [Offline preflight](OFFLINE_PREFLIGHT_R2.json) records the R2 slug, bootstrap cap and size, archive hash, mounts and output paths without contacting Kaggle.
- [Resource and quota plan](RESOURCE_AND_QUOTA_PLAN_R2.json) separates measured parquet hashing from the unmeasured prediction runtime and sets the 21,600-second quota gate.
- [Collection plan](OUTPUT_COLLECTION_PLAN_R2.json) fixes the R2 receipts, output inventory and streamed byte bound.
- [Provider preflight attempts and correction](PROVIDER_PREFLIGHT_R2_CORRECTION_R1.json) records the fresh quota and exact owner/source checks; the initial incomplete source-mount check is preserved in the linked superseded attempt.
- [CP2 asset READY receipt](ASSET_DATASET_READY_R2.json) records the private version-1 dataset and the three uploaded filenames. The remote listing does not expose hashes; the worker validates mounted bytes before prediction.
- [R2 bundle READY receipt](BUNDLE_DATASET_READY_R2.json) records private version 1 with the exact archive and manifest filenames. The kernel source directory was skipped by the dataset uploader and is pushed separately.
- [Root admission receipt](ROOT_ADMISSION_R2.json) binds the fresh typed quota, exact five private READY dataset versions and root authority before the kernel push.
- [Kernel push receipt](KERNEL_PUSH_R2.json) records private kernel version 1, the RUNNING state at first verification, exact provider-returned source hash and resource caps. [The preserved owner-list field gap](KERNEL_PUSH_OWNER_LIST_FIELD_GAP_R1.json) explains why nullable list fields were not treated as evidence against the successful push; the authenticated `GetKernel` response supplied the exact privacy/version fields.
- [First terminal inventory failure](COLLECTOR_INVENTORY_FAILURE_R1.json) preserves the R2 allowlist mismatch and confirms that no result body was downloaded.
- [Complete bounded remote inventory](REMOTE_OUTPUT_INVENTORY_R1.json) lists all 15 result paths and header lengths, binds the runner hash and shows the total is within the collection cap.
- [R3 collector preflight](COLLECTOR_PREFLIGHT_R3.json) binds the frozen R2 release receipts and runner archive, records the 500 MB CPU test results, and leaves the 3 GB collection unstarted pending root's fresh combined gate at that preflight point; the later terminal receipt records the completed collection.
- [R3 collector source](collect_terminal_r3.py) adds only the standard runner heartbeat with a size/schema bound; its focused [heartbeat tests](tests/test_collect_terminal_r3.py) cover the runner-generated schema, unexpected paths/fields and non-finite values. The [terminal collection receipt](TERMINAL_COLLECTION_R3.json) records successful validation; official F3 grading remains pending with root.
- [Collected small runner receipts, sidecars and logs](collected/attempt_002/) are committed with their hashes also bound by the terminal receipt. Large prediction NPY payloads remain at the same physical paths but are intentionally excluded from Git.
