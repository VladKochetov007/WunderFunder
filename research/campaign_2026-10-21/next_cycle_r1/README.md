# October21 next local/Kaggle cycle

The [accepted spec](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md)
records the user's new 30-local-GPU-hour cycle, staged complete-panel allocations,
six-fit i1 screen and four-fit dynamics screen. B300 is deferred. No gain or fit
completion is implied by this packet.

- [Frontier](FRONTIER.md): actual implementation, recovery, resource accounting
  and next actions, owned by root.
- [Public receipt](PUBLIC_RECEIPT_R1.json): user-reported latest package score and
  runtime versus TL3; exact package identity remains to be verified from artifacts.

## First recovery findings

[Tensor audit](WEIGHT_MEAN_COMPATIBILITY_R1.json) blocks the exact frozen mean: parent head23 versus child25 rows; no averaging or new fits. [Quota](KAGGLE_QUOTA_R1.json) is 2,345.378 seconds, zero provider reserved, refresh October24. [Native recovery admission](NATIVE_RECOVERY_ADMISSION_R3.json) and [finite sequence](NATIVE_RECOVERY_SEQUENCE_R3.json) repair the NPY-header guard after three passing encoding checks; unchanged CP5/SWA packages will be replayed and graded, not rebuilt. CPU-only, no local GPU charge.

- [I1 profile admission/recovery](I1_PROFILE_ADMISSION_R2.json), [source hashes](I1_PROFILE_SOURCE_R1.json), [R1 plan](I1_PROFILE_SEQUENCE_R1.json) and [failure report](I1_PROFILE_REPORT_R1.json), [R2 plan](I1_PROFILE_SEQUENCE_R2.json): three synthetic invariant tests pass; original redundant width flags failed CLI validation before data, corrected to equivalent inherited widths. Sustained110batch profile admitted; no scientific fit. Raw logs `i1_profile_r1.log`, `i1_profile_r2.log` preserve attempts.

- [Budget ledger](BUDGET_R1.json): explicit stage ceilings, per-arm fits, reused controls and teachers; completed charges versus active profile reservations. [R2 profile report](I1_PROFILE_REPORT_R2.json) records a second CLI-only rejection: profile early exits cannot save training state. [R3 finite profile](I1_PROFILE_SEQUENCE_R3.json) removes that flag only for profiling; scientific fit recipes retain it.

- [Sustained profile result](I1_SUSTAINED_PROFILE_R3.json) and [terminal report](I1_PROFILE_REPORT_R3.json):1,100finite updates,4.36124s/batch,0OOM, GPUallocated0.98GB/reserved1.22GB;3GBcgroup had file-page reclaim. Provisional full-panel cost4.85h includes20%fitmargin/setup/export reserves; final command review still pending. Engineering evidence, not quality.

- [i1 six-fit packet](i1/README.md), [admission/source seal](I1_PANEL_ADMISSION_R1.json), [R2 sequence](I1_PANEL_SEQUENCE_R2.json), and [launcher repair](I1_PANEL_LAUNCH_REPAIR_R2.json): matched two-cell A/B/C implementation, eight passing CPU checks, sustained thermal profile, and live bounded execution. R1 had an omitted loader setting and startup OOM; preserved, charged, repaired explicitly. No quality result yet.
- [Dynamics review](DYNAMICS_REVIEW_R1.json): independent source audit passes the frozen recipe, with setup-cost and general-validator limitations; four CPU tests pass. [Canonical implementation/recipe](../../experiment_drafts/dynamics-aux-20261021/README.md) has no completed profile/fit.
- [B16 offline recovery](../kaggle_h512_b16_offline_recovery_r1/README.md): separate source-pinned saved-weight mean/export/grade adapter, preserving missing initialization digest and original failed acceptance. Independent review pending; zero new fits.

- [Memory-safety correction](MEMORY_SAFETY_CORRECTION_R1.json): user warning, exact R1kill diagnosis and correctedR2/native counters. Additional heavy admissions held while both active; host pressure remains a material constraint.

- [B16 independent review](B16_RECOVERY_REVIEW_R1.json): identified missing metadata byte binding, fixed with exact provenance/historySHA pins verified against terminalcollectionmanifest; reviewer confirmed resolution. Remaining source/mean/coverage/fixedgrade checks passed statically; execution pending.
