# Age R3 Kaggle CPU profile handoff

## Current execution and checksum correction

Root's [R2 scope amendment](PROFILE_SCOPE_AMENDMENT_R2.json) explicitly admits an engineering microprofile: A/G/F heads, three epochs on 16 fitting sequences, 360 optimizer steps total, eight target-free evaluation sequences, no quality grade or saved scientific endpoint. The [R2 driver](kernel_driver_template_r2.py) enforces one CPU and the implemented 4 GB sampled RSS watchdog, not remote cgroup containment. The preparation description below remains historical.

Kernel version 1 failed after 0.0286 worker seconds, before extraction, imports, fitting or data reads. The [receipt](PROFILE_DRIVER_RECEIPT_FAILED_R3.json) and [log](KERNEL_FAILED_R3.log) preserve the error. Root traced it to a **65-character copied overlay-manifest checksum** in the handoff/admission. Earlier preparation/review receipts contain that typo and do not establish successful execution. The actual checksum is `138dafd2f03e500190c7820a0aac86a459a32caa7399d2e4421e7e81b2364659`.

The [R4 repair](REPAIR_AND_PREFLIGHT_R4.json), [actual overlay fixture](ACTUAL_OVERLAY_FIXTURE_R4.json) and [root admission](PROFILE_F0_CPU_R4_ADMISSION.json) derive the checksum from bytes. The exact rendered driver accepts the admission, its normalized hash matches, and real overlay application plus full source-map verification pass against 43 small base-archive members. The fixture did not import the model or read array/target payloads.

[Dataset version log](DATASET_VERSION_R4.log) and [kernel push log](KERNEL_PUSH_R4.log) record version 2, running at 11:33:35 UTC. Its actual ID is [`vladkochetov07/wnn-age-readout-f0-cpu-profile-20261024-r3`](https://www.kaggle.com/code/vladkochetov07/wnn-age-readout-f0-cpu-profile-20261024-r3), derived by Kaggle from the title. At that pre-retry snapshot, root owned collection under the 900-second worker ceiling; cache parity and import success were unmeasured. The later operator collection and version-3 cache-membership failure are recorded below.

## Latest retry outcome — 2026-10-24 11:49 UTC

The prior R4 version-2 attempt ended with Kaggle `ERROR`, a downloaded empty
kernel log, and no driver receipt. Its worker/quota usage remains **unknown**;
the operator did not count it as zero and reserved the full 900 seconds for
one exact retry. The [startup-failure record](STARTUP_FAILURE_R4.json) keeps
that event separate from the later retry.

Root authorized one same-code, same-input retry. The
[pre-push admission](PROFILE_F0_CPU_RETRY_R1_ADMISSION.json) records kernel version 3, unchanged
dataset slugs/source and a 900-second, one-CPU, 4-GB RSS-watchdog, zero-swap
profile budget. The [push log](KERNEL_PUSH_RETRY_R1.log) records successful
submission. No new dataset version or source was uploaded.

Version 3 terminated with Kaggle `ERROR`, but this time the filtered terminal
download included a driver receipt. Archive extraction and overlay/source/raw
identity checks passed. The profile subprocess then failed before opening the
portable target cache: `ValueError: portable fit cache subset membership
identity changed`. The fit sidecar's `subset_members_sha256` did not match the
frozen subset metadata's `selected_members_sha256`. The receipt reports 36.66
seconds total, 18.07 seconds in the profile subprocess, 429,977,600-byte peak
process-tree RSS, one CPU, and zero swap. No profile microfit or optimizer
step completed; no scientific fit or quality score was produced. This is a
cache-membership identity failure, not a numerical result or a repeat of the
empty-log startup failure.

The [terminal receipt](TERMINAL_RETRY_R1.json), [final provider status](KERNEL_STATUS_DETAIL_RETRY_R1.json), and [collection log](COLLECTION_RETRY_R1.log) preserve the exact version-3 outcome. Small filtered
outputs and the exact admission sidecar are under
[`terminal_retry_r1/`](terminal_retry_r1/); the 1.14-GB input archive and all
cache payloads were excluded from collection. No further retry was made;
root owns any cache repair or new admission.

The source-level schema repair for this mismatch passed an independent
[metadata-only review](MEMBERSHIP_REPAIR_INDEPENDENT_REVIEW_R1.md) using the
actual fold-0/fold-1 producer sidecars. It keeps the exact fit/evaluation
membership equality gate and rejects a changed digest; remote reconstruction
and runtime remain unverified.

The terminal receipt contains a manually copied 65-character overlay-manifest
SHA field. Preserve that original record, but use the separately derived
[terminal receipt errata](TERMINAL_RETRY_R1_ERRATA.json): it hashes the
physical manifest and confirms the collected driver receipt and admitted
sidecar contain the matching 64-character digest. This is a reporting typo;
the provider's manifest/source checks passed.

## Portable membership repair

The version-3 failure was traced to a consumer schema mismatch: raw v1 subset
JSON stores `selected_members_sha256` at the top level, while the portable
consumer incorrectly looked for a nested `manifest` object. The frozen source
subset and all six source-produced fold sidecars agree on their membership
digests. The focused fix and two-fold actual-metadata fixture are recorded in
[`membership_repair_r1/README.md`](membership_repair_r1/README.md), with exact
recipe, producer, consumer and sidecar pins. Both fold metadata paths pass and
a changed membership digest remains rejected. No cache payloads, labels, or
arrays were read; no Kaggle upload or retry has occurred. Root owns any new
overlay, admission and remote profile.

Additional records: [R2 recipe](KERNEL_RECIPE_R2.json), [staging](STAGING_R2.json), [original admission](PROFILE_F0_CPU_R3_ADMISSION.json), [dataset creation](DATASET_CREATE_R3.log), [initial kernel push](KERNEL_PUSH_R3.log) and [collection](COLLECTION_R3.log). The [PR26/27 feasibility analysis](FEASIBLE_NEXT_SCIENCE_R1.md) estimates that a p10/p25 innovation panel exceeds two single-device hours on reported B300 timings. The [CUDA candidate](../kaggle_age_gpu_r1/README.md) remains unadmitted pending cache costs and independent review.

This packet prepares one bounded retry of the F0 CPU cache profile after the
R2 run stopped during import with `ModuleNotFoundError: wnn.i1_history`. R2
ended after 36.874 seconds, before a profile result, cache-quality grade or
scientific fit. R3 adds the missing source closure while keeping the recipe and
numerical implementation unchanged. The independent source review passed, but
the R3 import and remote profile remain unvalidated.

The profile is capped at 900 seconds, one CPU core, 4 GB process-tree RSS and
zero swap. It uses the existing private R2 suffix dataset and the existing
`wunder-funder` dataset. Only the 368,640-byte R3 overlay and a separately
root-issued admission need a new private dataset upload. The overlay expands to
322,258 bytes. The 1,143,552,000-byte R2 archive is not copied or rebuilt.
GPU, TPU, internet, fitting, target-cache access and quality scoring are off.
Root owns admission, provider uploads, execution and collection; no provider
mutation or remote run has occurred in this preparation.

## Frozen inputs and source closure

- R2 base archive: 1,143,552,000 bytes, SHA256
  `971871550f50be91c5afe1134e1012a3ba3b9c94e2157b4f8b6ed17a62442ef8`;
  its bundle manifest SHA256 is
  `ea241506eadd0a34f437c699bab3b7784a69b818b710e3d016c9b3599774f99e`.
- R3 actor source manifest: SHA256
  `6ce40d6e4980caf5d7b41e47f104e4a845a30f557c151d60c5ca3852024a1e5a`.
  It pins 66 source modules, up from R2's 28.
- R3 overlay archive: 368,640 bytes, SHA256
  `0b1e3c66bd76ae4f081a508b6bbbc33c31c81c29a80a71027754b7ddbfeabe93`;
  its manifest SHA256 is
  `138dafd2f03e500190c7820a0aac86a4599a32caa7399d2e4421e7e81b2364659`.
  The 40 members are the R3 manifest, 38 newly included source dependencies,
  and an updated `age_readout_provenance.py`. That replacement checks against
  its expected R2 SHA256 before applying. The only source change is the
  provenance `SOURCE_FILES` list; the machine recipe SHA remains
  `0a67fe12c520e56bcf14a3a1efecb5030f3aa7ce932101814c9dd1c675c930fd`.
- Existing export receipt says 60 bundle files / 1,143,486,298 expanded bytes,
  with no raw Parquet, target cache, full cache arrays or scientific fits.

The original [R2 terminal receipt](../publication_source_r1/reference_snapshots/37_KAGGLE_PROFILE_TERMINAL_R2.json), [R3 actor source map](../publication_source_r1/reference_snapshots/35_ACTOR_SOURCE_SHA256_R3.json), [R3 overlay manifest](../publication_source_r1/reference_snapshots/41_SOURCE_OVERLAY_MANIFEST_R3.json), [overlay preparation receipt](../publication_source_r1/reference_snapshots/42_SOURCE_OVERLAY_PREPARE_R3.json) and [independent R3 source review](../publication_source_r1/reference_snapshots/36_INDEPENDENT_SOURCE_REVIEW_R3.json) carry the underlying evidence. The overlay archive and manifest SHA values were rechecked from the existing 368,640-byte local overlay. `SOURCE_CLOSURE_REVIEW_R1.json` records these pins and the applied checks.
`kernel_driver_template.py` hashes both input archives, validates the overlay
manifest and every overlay member, checks absent paths and the single expected
replacement hash, applies the overlay, then rehashes the full R3 source map
before invoking the actor. This driver template has not been imported,
compiled, tested or run remotely in this task.

## Root-owned admission and upload

The R3 admission does not exist yet. Root must issue
`PROFILE_F0_CPU_R3_ADMISSION.json`, pinning the R2 base archive and manifest,
R3 overlay archive and manifest, R3 source manifest, unchanged recipe, driver
template, transport and export receipt. It must authorize exactly one CPU core,
at most 900 seconds, zero swap, `new_scientific_fits: 0`, and
`quality_scoring: false`. The finalizer must render only the
`__PROFILE_ADMISSION_SHA256__` marker in the driver template and pin the
normalized template hash recorded in `KERNEL_RECIPE_R1.json`.

After root authorization, stage the small overlay dataset with
`dataset-metadata.json`, `source_overlay_r3.wnn` and the root-issued admission.
Reuse the already-published R2 base and raw-data datasets listed in
`kernel-metadata.template.json`. Do not upload another copy of the base
archive. Root must inspect whether the proposed private overlay dataset and
kernel IDs already exist before choosing create versus version/push commands.
The exact metadata and commands are in `KERNEL_RECIPE_R1.json`; they were not
executed here.

The child command remains the existing profile-only CLI with `--fold 0
--device cpu`. The driver's 45-second finalization reserve means the profile
child stops by 855 seconds when the full 900-second allowance is used. A
successful result is only a runtime/import profile; it is not a scientific
fit or quality result. Record any import error as execution evidence and do
not grade model quality from it.

## Files

- `kernel_driver_template.py`: R3-aware copy of the reviewed R2 driver, with
  overlay verification and 900-second ceiling.
- `kernel-metadata.template.json`, `dataset-metadata.template.json`: private
  Kaggle metadata templates for the kernel and small overlay dataset.
- `KERNEL_RECIPE_R1.json`: exact input, resource, source and upload pins.
- `SOURCE_CLOSURE_REVIEW_R1.json`: source-delta and driver-closure review.
- [INDEPENDENT_REVIEW_R1.json](INDEPENDENT_REVIEW_R1.json): independent
  static review of the R1 recipe, R3 driver, base bundle and source overlay.
  Source pins, overlay payload hashes, archive guards, child command, privacy
  settings and the 900-second ceiling passed static review. Admission remains
  conditional on a one-core-only validator, an explicit engineering-microfit
  disclosure for the 360 optimizer steps on fitting targets, and aligning the
  admitted RAM value with the enforced 4,000,000,000-byte RSS watchdog. No
  remote imports or Kaggle run were performed.
- [INDEPENDENT_REVIEW_R2.json](INDEPENDENT_REVIEW_R2.json): follow-up on the
  root-owned R2 scope amendment, recipe and driver hashes. Status: conditional
  pass for root-controlled admission; the one-core, RAM and optimizer-bearing
  profile-scope findings are fixed. Preserve the admitted scope JSON beside
  the profile/driver receipts because the R2 runtime receipt does not repeat
  all micro-fit and training-label fields. Remote imports, execution and
  provider mutations remain unverified and root-owned.
- [FEASIBLE_NEXT_SCIENCE_R1.md](FEASIBLE_NEXT_SCIENCE_R1.md): PR26/27 short-screen source/command review and bounded paired-panel cost estimate; a p10/p25 comparison does not fit under two GPU-hours.
- [Age F0 CUDA engineering-profile candidate](../kaggle_age_gpu_r1/README.md): a separate strict-FP32 GPU adaptation of the canonical profile runner. It rebuilds and deletes the full cache, so root must use the R3 CPU cache-parity timing before considering its 600-second cap.

The [original engineering draft index](../publication_source_r1/reference_snapshots/08_README.md) links the broader source and resource records.


## Fresh CPU packet after the membership repair

A new [R4 CPU profile packet](../kaggle_age_r4/README.md) now binds the
independently reviewed two-fold membership consumer repair into a fresh
389,120-byte source overlay and updated admission template. It proposes a
single 900-second, one-core engineering cache/profile attempt and still has no
new scientific fit or quality score. The packet is not admitted or uploaded;
root owns the provider-ID check, budget and any launch.
