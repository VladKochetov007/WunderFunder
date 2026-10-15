# F3 teacher CP2 export packet R2

R2 is a quota-safe successor to the frozen R1 packet. It keeps the same verified F3 seed-3 H384 epoch-2 checkpoint, provenance, split, prediction scopes and 18,000-second export worker cap. The generated private kernel bootstrap enforces a 21,600-second outer wall limit, emits a matching timeout diagnostic, and leaves a 3,600-second startup/finalization margin. R1 files and hashes remain unchanged; R2 uses its own bundle and kernel slug so it cannot silently replace the prior packet.

The worker serially exports causal FP32 predictions for 10,539 fit-train, 1,368 fit-valid and 1,873 full-valid sequences. Their raw arrays total 2,204,800,000 bytes. It performs no fit, label-based score, grade or selection. Prediction runtime has not been measured; the 18,000-second worker bound preserves the three existing 6,000-second per-export caps. The 8 GB host reservation, 12 GiB CUDA allocator cap, 13 GiB observed stop and 6 GB free-work-disk gate are unchanged from R1.

The bundle slug is `wnn-f3-t3-cp2-exports-20261015-r2`, and the kernel id is `vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r2-runner`. The private CP2/provenance asset dataset identity stays `vladkochetov07/wnn-f3-t3-cp2-assets-20261015-r1`, whose local content-addressed stage is verified by the builder before packaging. No Kaggle provider action has occurred for either packet.

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

Only after an independent critic gives GO, root records a fresh typed GPU quota of at least 21,600 seconds remaining, confirms its owner inventory has no conflicting job, and admits this exact set of hashes may root run the following private operations. The reported quota snapshot at `2026-10-15T11:50:33Z` showed 26,627.389 seconds remaining; that observation is stale and does not authorize launch.

```sh
kaggle datasets create -p /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/kaggle_assets/wnn-f3-t3-cp2-assets-20261015-r1
kaggle datasets create -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/bundle_wnn-f3-t3-cp2-exports-20261015-r2
kaggle kernels push -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r2-20261015/research/campaign_2026-10-15/f3_teacher_export_r2/bundle_wnn-f3-t3-cp2-exports-20261015-r2/kernel
```

Root must verify both datasets are private and READY at the exact versions mounted by the kernel, and write `ROOT_ADMISSION_R2.json`, `ASSET_DATASET_READY_R2.json`, `BUNDLE_DATASET_READY_R2.json`, and `KERNEL_PUSH_R2.json`. The admission binds the 21,600-second outer cap and 18,000-second worker cap; the push receipt binds the positive kernel version, archive and metadata hashes, and private status. No automatic retry is authorized.

## Terminal collection

After the exact pushed kernel version is terminal, use the R2-only collector:

```sh
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-15/f3_teacher_export_r2/collect_terminal_r2.py \
  --repo-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

It requires the exact R2 admission and push receipts, inventories the fixed `results/` allowlist via GET headers before downloading, caps streamed collection at 2.5 GB, and writes to a unique physical-disk directory with a 4 GB free-space floor. A complete export claim requires terminal `COMPLETE`, all expected outputs, downloaded lengths and hashes matching both the worker manifest and prediction sidecars, the worker pipeline's CP2/provenance/split identity, FP32 shapes, and a bounded mmap scan confirming every prediction is finite. A failed post-download check is preserved in the terminal receipt and cannot be reported as complete. Root runs the official F3 grader afterward; this packet itself makes no teacher-quality claim.

## Navigation

- [Source and ancestry binding](SOURCE_BINDING_R2.json) pins CP2, provenance, source-qualified split, frozen predictor and generated R2 kernel artifacts.
- [Offline preflight](OFFLINE_PREFLIGHT_R2.json) records the R2 slug, bootstrap cap and size, archive hash, mounts and output paths without contacting Kaggle.
- [Resource and quota plan](RESOURCE_AND_QUOTA_PLAN_R2.json) separates measured parquet hashing from the unmeasured prediction runtime and sets the 21,600-second quota gate.
- [Collection plan](OUTPUT_COLLECTION_PLAN_R2.json) fixes the R2 receipts, output inventory and streamed byte bound.
