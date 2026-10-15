# F3 teacher CP2 export packet R1

This packet prepares a private, export-only Kaggle job for the already-fitted F3 seed-3 H384 teacher at epoch 2. The checkpoint, provenance sidecar, fold-3 split hash and training counters are bound to the collected R5 verification receipt. The predictor is the unchanged R1 source snapshot. The worker produces causal FP32 predictions for fit-train, fit-valid and full-valid membership; it does not fit, read labels for scoring, grade the teacher or select a model.

The serialized worker calls the frozen predictor three times in one Python process. This preserves the R1 command semantics while allowing its process-local portable-parquet digest cache to avoid repeating the measured 1,125.241-second source hash scan. The source-bound worker verifies both checkpoint and provenance hashes, checks F3 membership, validates each output sidecar and finite-scans the arrays in bounded windows before `kaggle_gpu_runner.py` copies declared outputs. Prediction time has not been measured. The 18,000-second cap is the sum of the three existing 6,000-second R1 export caps, not a runtime estimate.

The raw prediction payload is 2,204,800,000 bytes across 10,539 fit-train, 1,368 fit-valid and 1,873 full-valid sequences. The inherited T4 export limits are 8 GB host RAM, a 12 GiB allocator cap and a 13 GiB observed stop; the worker requires at least 6 GB free working disk before exporting. These are prepared bounds, not proof of measured export capacity. The latest quota snapshot in [the resource plan](RESOURCE_AND_QUOTA_PLAN_R1.json) is operator-reported and must be replaced by a fresh typed quota observation for root admission.

## Frozen inputs and commands

The checkpoint is staged, without changing the source file, at `/home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/kaggle_assets/wnn-f3-t3-cp2-assets-20261015-r1`. Its content-addressed manifest includes `model_ep2.pt` and `provenance.json`; the uploaded dataset should remain private and must not be versioned after release. The generated bundle slug is `wnn-f3-t3-cp2-exports-20261015-r1`, producing kernel id `vladkochetov07/wnn-f3-t3-cp2-exports-20261015-r1-runner`.

After root reviews the source packet and grants a fresh provider admission, the local commands are:

```sh
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-15/f3_teacher_export_r1/prepare_export_packet_r1.py \
  --root /home/vlad/development/wunder/worktrees/f3-teacher-export-r1-20261015 \
  --source-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --rebuild-local

kaggle datasets create -p /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/kaggle_assets/wnn-f3-t3-cp2-assets-20261015-r1
kaggle datasets create -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r1-20261015/research/campaign_2026-10-15/f3_teacher_export_r1/bundle_wnn-f3-t3-cp2-exports-20261015-r1
kaggle kernels push -p /home/vlad/development/wunder/worktrees/f3-teacher-export-r1-20261015/research/campaign_2026-10-15/f3_teacher_export_r1/bundle_wnn-f3-t3-cp2-exports-20261015-r1/kernel
```

Those provider commands are recorded for root-operated use only; this actor has made no Kaggle API call, dataset creation, upload, version, or push. Before root executes them, it must check fresh typed GPU quota, the owner inventory, private dataset READY versions, exact bundle/kernel metadata hashes, and an aggregate time budget that covers the 18,000-second worker plus finalization. No automatic retry is authorized.

## Output and collection

The job spec [export job](jobs/export_f3_teacher_cp2_triplet_20261015_r1.json) declares the six prediction arrays/sidecars and the worker pipeline receipt. [The collection plan](OUTPUT_COLLECTION_PLAN_R1.json) pins every expected remote `results/` path, a 2.5 GB total inventory bound, GET-header size inventory before downloading, a 4 GB physical-disk floor and one streaming collection attempt. It reuses the reviewed R4 streaming helper by hash but does not reuse the old R5 collector, whose kernel id and 1 GB cap target another job.

After the exact pushed kernel version is terminal, and after root records the admission and push receipts, root should run:

```sh
/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python \
  research/campaign_2026-10-15/f3_teacher_export_r1/collect_terminal_r1.py \
  --repo-root /home/vlad/development/wunder/wnn_connectome_starterpack
```

The collector requires root admission and push receipts binding `kernel_id`, `bundle_archive_sha256` and `kernel_metadata_sha256`; admission also records `status=admitted` and the 18,000-second cap, while the push receipt records the positive `kernel_version` and `private=true`. It checks the helper source hash, requires a physical filesystem and 4 GB free disk, inventories only the fixed `results/` allowlist by GET headers, then streams each file once into the unique collection directory. The total cap is 2.5 GB. A completed kernel must have a complete worker result, no missing outputs, a completed three-stage pipeline receipt, exact checkpoint/provenance/split identity, FP32 shapes, matching sidecar hashes and finite predictions. Errors remain execution evidence and are not retried automatically. Scoring is a separate root-owned step using the official frozen grader; this packet makes no teacher-quality claim.

## Navigation

- [Source and ancestry binding](SOURCE_BINDING_R1.json) records the exact checkpoint, training provenance, source predictor, job and generated bundle hashes.
- [Offline preflight](OFFLINE_PREFLIGHT_R1.json) records the generated title-derived slug, bootstrap size, archive hash, external mounts and declared outputs without contacting Kaggle.
- [Resource and quota plan](RESOURCE_AND_QUOTA_PLAN_R1.json) distinguishes the inherited cap and measured parquet-hash scan from the unmeasured three-export runtime.
- [Output collection plan](OUTPUT_COLLECTION_PLAN_R1.json) defines the terminal inventory, byte budget and post-download gates.
