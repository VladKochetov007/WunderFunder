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

## Six-fit i1 panel admitted and running

The [i1 packet](i1/README.md), [source-sealed admission](I1_PANEL_ADMISSION_R1.json) and [R2 finite sequence](I1_PANEL_SEQUENCE_R2.json) now own six fits (two per A/B/C arm), zero teacher fits, no reused fit controls. Source is clean isolated b1b680d4a, excluding later dynamics changes. Prior H512 full1873sequence exports took294–397seconds; this panel exports443/468groups, supporting the reserved1800seconds for all exports/grade. Projection remains17,452/18,000seconds, not a runtime guarantee.

R1 launcher omitted the profiled zero-worker loader setting and failed at startup with a cgroupOOM; no checkpoint/fit result. [Repair receipt](I1_PANEL_LAUNCH_REPAIR_R2.json) preserves the12second charge and explicit synchronous-loader correction. R2 `wnn-next-i1-panel-r2-20261021.service`, invocation`b1cf6f061ac04873a9f5cf3a56333e44`, is live at1CPU/3GB/noSwap/17,533secondcap. Log confirms workers0/prefetch0 and4,620scheduledF0updates. Same source/recipe/batch, strictFP32. Root owns monitoring and results; no quality score yet.

## Dynamics and offline recovery implemented

[Independent dynamics review](DYNAMICS_REVIEW_R1.json) passes the frozen four-fit recipe; four root CPU tests pass. Full target-statistics preprocessing must enter the upcoming cost profile. See [implementation/recipe](../../experiment_drafts/dynamics-aux-20261021/README.md); code is integrated, profiles/fits pending and optional features default off. [B16 offline recovery](../kaggle_h512_b16_offline_recovery_r1/README.md) is integrated and undergoing independent review. It preserves the absent historical initial tensor digest rather than inventing it. No additional Kaggle launch.

Native CP5/SWA replay/grade remains live under1CPU/1.8GB. The root `scripts/package_submission.py` temporarily retains its pre-dynamics bytes (SHA51468297...) because both replay and grader bind that historical exporter. Restore its committed HEAD version only after the entire native sequence finishes; actual ZIPs are unchanged.

## User memory steering

User warned about killed processes. [Safety correction](MEMORY_SAFETY_CORRECTION_R1.json) identifies the two R1loader-launch kills, records correctedR2/native OOM0, and closes additional heavy/test/GPU admission until native completes. R2has reachedbatch40 with synchronousloading. Hostavailable~6.83GB and hostswap almostfull; taskswap remains0. Do not kill unrelated processes or useOOMto sizejobs.

The one-off `wnn-next-native-source-restore-20261021.service` / invocation`4f484f67a5784ba3bfbd005050304805` now owns restoring the temporarily pinned exporter after exactnativeinvocationtermination. It checks old/newSHA beforewrite, uses50MB/CPU5%/swap0 underparent2cores, and cannot launch computation. See [script](finish_native_source_restore.sh) and [log](native_source_restore.log). This keeps cleanupdurable while longfitscontinue.

## First i1 epoch and checkpoint verified

[I1 F0/B epoch0](I1_FIRST_EPOCH_R1.json) completed635.1seconds/1,540finiteupdates, zero skips/nonfinitegradients. Both8.8MBcheckpoint and27MBtrainingstate saved; post-saveOOM0. Epoch1isactive. CP2remainsfixed, no interimqualityselection. No additionalheavyjobadmitted.

[Draft PR #33](https://github.com/VladKochetov007/WunderFunder/pull/33) publishes the cycle implementation/recovery handoff against research-team. [Publication receipt](PUBLICATION_R1.json) records the isolated merge preserving both histories; no branch integration or quality promotion. Active fits keep their frozen source. Generated-tree refresh and merged-source checks remain pending a spare CPU slot.
