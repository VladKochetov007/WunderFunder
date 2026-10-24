# Age F0 CUDA profile candidate R2

Prepared 2026-10-24 12:09:34 UTC. This candidate includes the independently verified cache-membership repair and reconciles the allocator cap with the 4,000,000,000-byte recipe limit. It remains unadmitted. No Kaggle provider mutation, upload, import, cache read, profile, fit, or quality scoring was performed during preparation.

## Combined source changes

The candidate retains the two CUDA profile-only changes from the R1 source packet (CLI requires exactly one visible CUDA device; runner permits CUDA only for cache reconstruction in profile mode). The overlay also carries the repaired top-level membership digest reader and deferred Torch import. The new runtime module uses CUDA_ALLOCATOR_LIMIT_BYTES = 4_000_000_000; the previous source used 4 * 1024**3 (4,294,967,296 bytes). The runtime replacement is candidate payload at [source_delta/age_readout_runtime.py](source_delta/age_readout_runtime.py), not a mutation of the canonical working tree. No model, objective, optimizer, fit code, or machine recipe changed. Strict FP32 and TF32/autocast off remain pinned by the existing fit path.

## Proposed profile scope

F0/seed1001, 16 fit and eight target-free evaluation sequences; three engineering microfits (A/G/F), three epochs each, 360 optimizer updates total. Fitting labels are read to perform this engineering profile. Evaluation targets/masks are not read. It produces zero scientific fits, zero official metric calls and zero quality scores.

The intended resource envelope is one Kaggle GPU, one CPU core, 600 seconds total (45 seconds reserved for finalization), process RSS watchdog at 4,000,000,000 bytes, zero swap, no internet or TPU. This profile reconstructs and hashes the full portable cache in the same kernel; it does not reuse a cache from the CPU driver, which deletes its cache. The F0 cache consists of three source-verified payloads: fit sequences 11,568,960,128 bytes; evaluation sequences 4,380,480,128 bytes; and parent evaluation predictions 74,880,128 bytes. Their total is 16,024,320,384 bytes, as recorded in the [local cache verification receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/experiment_drafts/b300-fast-local-kaggle-20261023/LOCAL_CACHE_VERIFICATION_R1.json). Provider free disk capacity is not assumed: the candidate driver checks actual free bytes before work against extracted bundle bytes + the three-file F0 cache total + a 1,000,000,000-byte output reserve, with a 20,000,000,000-byte saved-output ceiling. The cache reconstruction/profile wall cost is still unknown, so the GPU candidate is not ready for root admission until the CPU retry supplies timing or root otherwise resolves that cost gate.

## Actual packet identities

- Overlay archive: 419,840 bytes, SHA256 3347d9ee699e81ae5281a9dbca6476f7f9f6ba5c903919033c7e20382b5e6b8d.
- Embedded/sidecar manifest SHA256 415b5ecad01b955dcadda10a3c667d621cf0e347d6e995ae7447428c2c054c25; 45 payload files and 46 total tar members.
- Source manifest SHA256 2a5c006812a363f111db27c8ce2f49ce751c8aabc303c3ad59137e2bd7d3b934 across 66 modules; the runtime candidate SHA256 is 866ea97fae1ecd1598243e56fc23c112a01b9cf2b3cb70e73009f7ae3126eb6e.
- Driver template SHA256 fc923a9620376beddb3419198b3782d34869b539cbab674738f7aee21cc5ae95.
- Full overlay archive: /home/vlad/development/wunder/research_tmp/resume4h_20261024/kaggle_age_gpu_r2_upload/dataset_upload/source_overlay_cuda_r2.wnn; root must issue the current admission and verify provider IDs before any upload.

## Review files

- [KERNEL_RECIPE_GPU_R2.json](KERNEL_RECIPE_GPU_R2.json): full recipe and source pins, with cache-cost gate stated.
- [PROFILE_F0_CUDA_R2_ADMISSION.template.json](PROFILE_F0_CUDA_R2_ADMISSION.template.json): proposed fields only; not an admission.
- [SOURCE_SHA256_GPU_R2.json](SOURCE_SHA256_GPU_R2.json), [SOURCE_OVERLAY_MANIFEST_GPU_R2.json](SOURCE_OVERLAY_MANIFEST_GPU_R2.json), [SOURCE_CLOSURE_GPU_R2.json](SOURCE_CLOSURE_GPU_R2.json), [OVERLAY_BUILD_RECEIPT_R1.json](OVERLAY_BUILD_RECEIPT_R1.json), and [PACKET_STATIC_CHECK_R1.json](PACKET_STATIC_CHECK_R1.json): candidate closure and archive verification.
- [kernel_driver_template.py](kernel_driver_template.py), [kernel-metadata.template.json](kernel-metadata.template.json), and [dataset-metadata.template.json](dataset-metadata.template.json): root-review inputs; do not upload before admission.
- [source_delta/README.md](source_delta/README.md): pinned source snippets and their relationship to the canonical tree.

A profile run remains an engineering measurement, not model-quality evidence. This candidate is subordinate to the CPU cache-cost retry and has not been provider-authorized.
