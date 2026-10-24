# Age R4 Kaggle CPU membership-repair profile packet

Prepared 2026-10-24 12:09:34 UTC as a source-pinned candidate for root admission. Those preparation statements apply to that stage: no provider mutation, upload, import, cache-payload read, profile, scientific fit, or score occurred while the packet was being prepared. Root later admitted the packet and authorized one private CPU profile, whose terminal outcome is recorded below.

## Why this retry exists

The prior version-3 CPU profile verified the archive and source closure, then failed before cache reconstruction because the portable consumer expected the selected-members digest under a nested manifest object. The actual raw subset JSON stores it at top level. The repaired consumer reads the producer field and keeps exact equality between fit and evaluation cache-sidecar digests. The second small source edit defers a Torch-dependent import until full-cache validation so the metadata reader can run on a Torch-free fixture. See the [independent producer-metadata review](../kaggle_age_r3/MEMBERSHIP_REPAIR_INDEPENDENT_REVIEW_R1.md) and [repair source pins](../kaggle_age_r3/membership_repair_r1/SOURCE_PINS_R1.json).

## Exact candidate

- Machine recipe: F0/seed1001, unchanged SHA256 0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd.
- Engineering scope: 16 fit sequences, eight target-free evaluation sequences, three A/G/F microfits × three epochs, 360 optimizer updates total. It reads fitting targets to profile execution; it reads no evaluation targets or masks. It produces zero new scientific fits, no quality score, and no submission endpoint.
- Limits proposed for root admission: one CPU core; 900 seconds total; 45 seconds finalization reserve; 4,000,000,000-byte sampled process-tree RSS watchdog; zero swap. This watchdog is not Kaggle cgroup containment.
- Reuses the private 1,143,552,000-byte suffix dataset and the existing wunder-funder dataset; only the 389,120-byte source overlay plus root-issued admission would be uploaded.
- Proposed overlay dataset slug: vladkochetov07/wnn-age-cpu-overlay-20261024-r4. Proposed kernel identity remains vladkochetov07/wnn-age-readout-f0-cpu-profile-20261024-r3; root must verify existing versions and current campaign allocation before any action.

## Byte-derived pins

- Overlay archive: 389,120 bytes, SHA256 f3b21bdb171d689c1bdd3c8e052194cfc97ee9d915d2b4980bb74f6c66f6c097.
- Embedded/sidecar overlay manifest SHA256 20bf9d69eaab0e512be2e0aefb2d78a78d8b9d5fd4b3d3ab703e331ca3558bcb; 42 declared payload files and 43 total tar members.
- New 66-module source manifest SHA256 fb4f3080d907c4bdf6844e2e7588728f1e15c1c9be2f3ce14aa3b11f27d2e421.
- Driver template SHA256 b034c3b335e232440a317ad221c42d3518e80da852348d16a5592f14c9ff607f.
- Portable validator SHA256 a52a4a09bd57f321b89d241f5fad96d6ed6cbe889020187dbba5a6bf55ae21d6; portable identity module SHA256 e976c0311e53ce3d022e4ed6646dc28b83a137d18a136f0c3844476a8ec382c5.
- Full archive path: /home/vlad/development/wunder/research_tmp/resume4h_20261024/kaggle_age_cpu_membership_r4/dataset_upload/source_overlay_cpu_r4.wnn. Physical scratch keeps upload-only artifacts out of the repo. A clean upload staging folder with the private dataset metadata and overlay, but no admission file, is at /home/vlad/development/wunder/research_tmp/resume4h_20261024/kaggle_age_cpu_membership_r4/dataset_upload. See [UPLOAD_STAGING_R1.json](UPLOAD_STAGING_R1.json). The exact byte manifest and recipe are linked below.

## Files and admission status

- [KERNEL_RECIPE_CPU_R4.json](KERNEL_RECIPE_CPU_R4.json): full source, profile, resource and proposed command pins. Root should review and issue a separate current admission.
- [PROFILE_F0_CPU_R4_ADMISSION.template.json](PROFILE_F0_CPU_R4_ADMISSION.template.json): prepared fields only; it is not admitted and must be replaced by root's current admission.
- [SOURCE_SHA256_CPU_R4.json](SOURCE_SHA256_CPU_R4.json), [SOURCE_OVERLAY_MANIFEST_CPU_R4.json](SOURCE_OVERLAY_MANIFEST_CPU_R4.json), [SOURCE_CLOSURE_CPU_R4.json](SOURCE_CLOSURE_CPU_R4.json), [OVERLAY_BUILD_RECEIPT_R1.json](OVERLAY_BUILD_RECEIPT_R1.json), [PACKET_STATIC_CHECK_R2.json](PACKET_STATIC_CHECK_R2.json), and [manifest-copy erratum](SOURCE_OVERLAY_MANIFEST_ERRATA_R1.json): exact source closure and member-by-member archive verification.
- [kernel_driver_template.py](kernel_driver_template.py), [kernel-metadata.template.json](kernel-metadata.template.json), and [dataset-metadata.template.json](dataset-metadata.template.json): staged recipe inputs; do not push these until root admission.
- [UPLOAD_STAGING_R1.json](UPLOAD_STAGING_R1.json): isolated local overlay/metadata staging is ready, but contains no root admission and has not been uploaded.

The later root admission bound these bytes, the 900-second budget, and provider identities. Earlier CPU failures remain separate historical records; the admitted R4 outcome is recorded in the final section below.


## Manifest sidecar correction

Root's pre-admission check found that the repository sidecar had been copied
from an earlier 40-member preparation while the final 42-member archive
already contained the correct manifest. No admission or provider action had
occurred. The canonical [manifest sidecar](SOURCE_OVERLAY_MANIFEST_CPU_R4.json)
has now been replaced byte-for-byte from the embedded archive manifest
(8688 bytes, SHA256 20bf9d69eaab0e512be2e0aefb2d78a78d8b9d5fd4b3d3ab703e331ca3558bcb). The earlier sidecar bytes are
preserved in [SOURCE_OVERLAY_MANIFEST_CPU_R4_COPY_ERRATUM_R1.json](SOURCE_OVERLAY_MANIFEST_CPU_R4_COPY_ERRATUM_R1.json);
[the erratum](SOURCE_OVERLAY_MANIFEST_ERRATA_R1.json) records both hashes and
the cause. The archive and source payloads are unchanged. The superseding
[static check](PACKET_STATIC_CHECK_R2.json) confirms exact embedded/sidecar
byte identity and both membership-repair members.


## Root admission and execution preflight

Root has issued admission SHA256
bb9a5a3d92401f685719a70abd1bea1a27b8347e0803e1620b462185c1ab73d7.
The executable [preflight receipt](ADMISSION_PREFLIGHT_EXECUTABLE_R1.json)
renders the exact kernel, validates the admitted fields, calls the actual
driver overlay applier on a fresh source-only base fixture, and verifies all
66 source pins. It opened no training/cache payload and called no profile.

The authenticated read-only provider check is in
[PROVIDER_ID_PREFLIGHT_R4.json](PROVIDER_ID_PREFLIGHT_R4.json). At preflight,
the exact new private dataset slug was absent from owner search and the
existing suffix archive and raw data file names/sizes matched the admitted
inputs. The staged dataset contained exactly the metadata, overlay, and
admitted root file; the kernel staging contained only the normalized rendered
driver and metadata. Root then authorized the version-1 private dataset and
version-4 kernel recorded below.

The subsequent private dataset version 1 and kernel version 4 were created/pushed and are recorded in [DATASET_CREATE_R4.json](DATASET_CREATE_R4.json), [DATASET_CREATE_R4_STATUS_R1.json](DATASET_CREATE_R4_STATUS_R1.json), and [KERNEL_PUSH_R4.json](KERNEL_PUSH_R4.json). Version 4 reached Kaggle `ERROR` after the profile driver spent 855.52 seconds. Archive, overlay, full 66-module source map and raw-source checks passed; the spawned profile process then hit the admitted deadline. It emitted no profile result, so cache reconstruction completion and microfit step count are unknown. This is execution evidence only, with no quality grade or scientific fit result. The filtered [terminal receipt](TERMINAL_R4.json), [collected small outputs](terminal_r4/), and [cache/panel feasibility note](CACHE_RECONSTRUCTION_AND_PANEL_FEASIBILITY_R4.md) preserve the outcome and its limits. No automatic retry is authorized.
