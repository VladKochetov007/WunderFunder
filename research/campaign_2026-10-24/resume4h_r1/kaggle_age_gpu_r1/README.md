# Age F0 CUDA profile candidate

This folder contains a source-pinned, root-admission-gated candidate to measure
the existing age-readout profile on one Kaggle CUDA device. It reuses the
canonical `age_readout_screen.py` profile runner and its existing fit code. The
small source overlay changes only the CLI/runner guards needed to pass the
portable F0 cache to CUDA; it does not copy or change model, optimizer, loss,
batching, precision or scoring code.

## Scope and limits

- One fold, F0/seed1001; 16 fitting sequences and 8 target-free evaluation
  sequences.
- Three engineering readout microfits (arms A/G/F), each for 3 short epochs
  and 120 optimizer updates: 360 updates total. Training-sample labels are
  read to execute those updates; evaluation target and scoring-mask columns
  are not read.
- This is a runtime profile only: zero new scientific fits, zero official
  metric calls, zero quality scores, and no weights or prediction artifacts.
- Strict FP32 and deterministic CUDA: TF32, autocast and scaling are off. The
  canonical fit path enforces these settings; the existing runtime limits its
  CUDA allocator to 4 GB. The patched CLI requires exactly one visible CUDA
  device and rejects portable `fit-fold` mode.
- Total kernel wall cap is 600 seconds, including a 45-second finalization
  reserve, leaving at most 555 seconds for the child process. CPU use is one
  core, RSS is capped at 4 GB, swap/internet/TPU are disabled.

## Cache cost gate

This candidate **does not reuse** a cache from the separate R3 CPU profile:
that driver deletes its generated cache at finalization. The CUDA driver
reconstructs and hash-validates the full F0 fit/evaluation cache in the same
kernel before invoking the canonical profile. The pinned fit cache alone is
16,024,320,384 bytes. The Kaggle duration for that reconstruction and the CUDA
profile is unknown.

Do not upload or admit this candidate until root has the R3 CPU cache-parity
receipt and confirms that its measured cache-open/rebuild phase leaves enough
time for the CUDA profile under 600 seconds. If it does not, this packet is not
a feasible 600-second retry without another cache-delivery design. No cache is
preserved or uploaded by this driver.

## Frozen inputs and source delta

**Seed correction (2026-10-24):** the first prospective packet labeled this
F0 profile seed1201. The actual pinned machine recipe at
[`recipe.json`](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/recipe.json)
has SHA256 `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`
and declares F0/seed1001. [`KERNEL_RECIPE_R2.json`](KERNEL_RECIPE_R2.json)
corrects only that profile metadata. The original
[`KERNEL_RECIPE_R1.json`](KERNEL_RECIPE_R1.json) is retained as the erratum
record and must not be used for admission. The payload-level hash audit is in
[`GPU_PACKET_HASH_AUDIT_R1.md`](GPU_PACKET_HASH_AUDIT_R1.md).

- Base R2 archive: 1,143,552,000 bytes, SHA256
  `971871550f50be91c5afe1134e1012a3ba3b9c94e2157b4f8b6ed17a62442ef8`; reuse
  the existing private dataset. Its pinned bundle manifest is
  `ea241506eadd0a34f437c699bab3b7784a69b818b710e3d016c9b3599774f99e`.
- Combined R3-plus-CUDA source overlay: 389,120 bytes, SHA256
  `8d5c2567589876855256e7009a5be23a9f4b56ff366956da072daf21a49b2716`;
  overlay manifest SHA256
  `c977ca865e2e40da4fe7f38d7ab53c533edd8f85dfd4df6266ea2c1a581e3fef`.
  It has 42 pinned file members plus the manifest and retains the 38-module
  R3 import-closure repair.
- Final 66-module source manifest SHA256
  `bf034ffc8ac0212f5a52d7d5b9c57a11bb54750275147aa70d601c0a7b9795fc`;
  the unchanged machine recipe is
  `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`.
- Only two existing source modules changed from R3: `age_readout_cli.py`
  permits portable paths only for profile mode and requires one visible GPU;
  `age_readout_runner.py` permits portable-cache reconstruction for the
  profile. The fit-fold path remains CPU-only. The exact file pins and review
  status are in [SOURCE_CLOSURE_GPU_R1.json](SOURCE_CLOSURE_GPU_R1.json); the
  manifest and source modules are included in the overlay.
- The R3 base archive and raw data stay mounted read-only. The only new upload
  would be the small overlay plus the root-issued admission. No provider calls
  or upload staging were performed.

## Files

- [KERNEL_RECIPE_R2.json](KERNEL_RECIPE_R2.json): corrected F0/seed1001
  source pins, profile
  counts, resource limits, input IDs and root-gated Kaggle commands.
- [KERNEL_RECIPE_R1.json](KERNEL_RECIPE_R1.json): preserved first draft with
  the incorrect seed label; superseded by R2.
- [GPU_PACKET_HASH_AUDIT_R1.md](GPU_PACKET_HASH_AUDIT_R1.md): hash checks made
  against the archived overlay bytes and current source payloads, with the
  external base-bundle verification limit stated.
- [INDEPENDENT_REVIEW_R1.md](INDEPENDENT_REVIEW_R1.md): independent static
  source/recipe review. It verifies the seed, source delta, target-free profile
  path and profile/resource scope, and identifies a 4.0 GB versus 4 GiB CUDA
  allocator cap mismatch plus the unresolved full-cache timing gate. No
  Kaggle execution or target reads were performed.
- [PROFILE_F0_CUDA_R1_ADMISSION.template.json](PROFILE_F0_CUDA_R1_ADMISSION.template.json):
  root admission fields embedded by the driver; status remains unissued.
- [kernel_driver_template.py](kernel_driver_template.py): R3-aware archive
  verification, single-GPU profile invocation, 600-second watchdog and cleanup.
- [source_overlay_cuda_r1.wnn](source_overlay_cuda_r1.wnn) and
  [SOURCE_OVERLAY_MANIFEST_GPU_R1.json](SOURCE_OVERLAY_MANIFEST_GPU_R1.json):
  exact R3-plus-CUDA source overlay and its pin manifest.
- [SOURCE_SHA256_GPU_R1.json](SOURCE_SHA256_GPU_R1.json): final actor source
  closure with unchanged numerical implementation.
- [kernel-metadata.template.json](kernel-metadata.template.json) and
  [dataset-metadata.template.json](dataset-metadata.template.json): private
  one-GPU kernel and small overlay dataset templates.
- [FEASIBLE_NEXT_SCIENCE_R1.md](../kaggle_age_r3/FEASIBLE_NEXT_SCIENCE_R1.md):
  separate PR26/27 feasibility review; this engineering profile does not run
  that scientific follow-up.

Root owns the admission, dataset/kernel existence checks, upload, execution,
collection and any later scientific-panel authorization. Independent source
review and all remote validation are pending. This packet has not been
imported, tested or run.


## R2 candidate correcting the source cap

The [combined CUDA R2 candidate](../kaggle_age_gpu_r2/README.md) retains the
profile-only CUDA guards, includes the membership repair, and caps the CUDA
allocator at exactly 4,000,000,000 bytes. The older R1 review and receipts are
preserved. This candidate is still unadmitted because full-cache reconstruction
and profile timing remains unknown; no upload or provider run occurred.

## Cache accounting correction

The earlier sentence above calling 16,024,320,384 bytes the “fit cache alone”
was incorrect. The local source receipt
[LOCAL_CACHE_VERIFICATION_R1.json](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json)
shows three F0 files: fit sequences 11,568,960,128 bytes, evaluation
sequences 4,380,480,128 bytes, and parent evaluation predictions 74,880,128
bytes. Their **total** is 16,024,320,384 bytes. The frozen earlier recipe and
terminal receipts remain unchanged; [the R2 candidate packet](../kaggle_age_gpu_r2/README.md)
uses the corrected names and records the driver's runtime free-disk check.
