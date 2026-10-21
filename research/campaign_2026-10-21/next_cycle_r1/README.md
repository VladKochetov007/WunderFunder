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
