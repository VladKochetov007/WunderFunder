# Next cycle frontier

Owner root; accepted October21. The [spec](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md)
is the current authority: 30 new local GPU-hours, plus verified Kaggle quota;
local/Kaggle only. Historical prior-cycle usage stays in its original receipts.

## Entry inspection

Local GPU idle (0% utilization, 75MiB, 73°C). Parent `wnn-campaign.slice` reports
CPU quota 2 cores, MemoryMax 5,000,000,000, MemorySwapMax 0. Existing H512 fit and
postfit are exited; native replay failed. Dirty existing logs/ledgers/mean proposal
are preserved. No new fit admitted. New-cycle GPU use: zero at entry.

## Active work

- Root: preserve contract; execute fixed saved-checkpoint compatibility/mean,
  recover predictions/grades and admit sustained i1 profile after checks.
- `i1_recipe` Luna/max: isolated code-only two-cell recipe/membership preparation
  and prefit causality/mask/restore review; no GPU authority.
- `dynamics_builder` Luna/max: isolated canonical auxiliary implementation and
  focused CPU tests; no GPU authority.
- `recovery_audit` Luna/max: read-only Kaggle/native failure and coverage review.

No new scientific result, full panel reservation, package or provider launch yet.

## First recovery findings

[Tensor audit](WEIGHT_MEAN_COMPATIBILITY_R1.json) blocks the exact frozen mean: parent head23 versus child25 rows; no averaging or new fits. [Quota](KAGGLE_QUOTA_R1.json) is 2,345.378 seconds, zero provider reserved, refresh October24. [Native recovery admission](NATIVE_RECOVERY_ADMISSION_R3.json) and [finite sequence](NATIVE_RECOVERY_SEQUENCE_R3.json) repair the NPY-header guard after three passing encoding checks; unchanged CP5/SWA packages will be replayed and graded, not rebuilt. CPU-only, no local GPU charge.

## Live execution

Native replay unit `wnn-next-native-recovery-r3-20261021.service`, invocation `676380f555cf4f1daae236b9eeb432a6`, CPU1/RAM1.8GB, exact CP5 then SWA and grade. I1 three synthetic tests passed; [R2 profile](I1_PROFILE_ADMISSION_R2.json) now runs under1CPU/3GB after R1CLI-only failure (2.125seconds). Root owns inspection and full-panel admission; no fit has started.

R2 also exited before data (1.832seconds): `--save-training-state` is invalid for a profile early exit. R3 removes it only from profiling; inherited H512width and CP2scientific recipe unchanged. Active `wnn-next-i1-profile-r3-20261021.service`, invocation `b5c000953acc44ac9f16441554a69f88`, now reached actual training batches, 367inputs/2.29Mparameters. Both CLI failures are preserved and charged in [budget](BUDGET_R1.json). Native guard independent review passed (`recovery_audit`, Luna/max); no quality conclusion yet.

## Sustained profile complete

[R3](I1_SUSTAINED_PROFILE_R3.json) passed,1100finiteupdates,4.36124s/batch,0OOM. Total profile charges454.953s. Complete six-fit stage provisionally17452s including profiles,20%fitmargin,360ssetup and1800sexport/grade reserve; no panel admitted until malformed amendedargv and evaluation-only grading are checked. CPUtests now allowed serially in spare1CPU/1.8GB; native replay continues.
