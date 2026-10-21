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

## Grader review amendment before quality outputs

[Independent packet review](DYNAMICS_PACKET_REVIEW_R1.json) passed exactplans, producer/consumer coverage and pairedstatistics. Root addedthe existing immutabletargetcacheSHA to both i1/dynamics graders andchecks itbefore/aftergrading; reviewerconfirmedresolution. No metric/blend/fitsourcechange and noqualityresultseen. Dynamics finitepacket integrated as b7ac5e5c5; runtime/profile/fits remainpending.

The tiny source-restoration observer was restarted alone as invocation`ded13f33b9164006ac063e02ecb5d8ed` after adding the successful-transient-unit collection case: if systemd has removed the unit, it requires all three native steps complete/returncode0 before restoring. No model/native worker was restarted. Shell syntax and missing-unit query behavior checked; observer uses about1MB RAM.

## Native recovery complete; i1 first control in final epoch

[Recovered native grade](NATIVE_GRADE_R3.json): originalTL3 0.70181812; CP5replacement 0.69445462 (−0.00736350); SWA345replacement 0.69444908 (−0.00736904). SWA−CP5 −0.00000553, interval[−0.00029533,+0.00026122]. Bothfullreplays/grade succeeded and temporaryexporterpin restored. Evidence is exposedF0fitteddiagnostic, unknownretainedTL3ancestry; no promotion. CPU-only native work charged0GPUseconds. I1F0/B has3,080completedupdates andlastobservedfinalepochbatch120/154; correctedworker OOM0/skips0. No complete paired screen result yet. No newheavyjobadmittedbythisstatuscheck.

## User-directed interpretation and next diagnostics

The user requested continued research and reconsideration of overlooked explanations. Root froze [the no-fit diagnostic plan](ENDPOINT_DIAGNOSIS_PLAN_R1.json); two independent source-only reviews found no mask, order, wrapper, or native-backend mismatch. R3 closes the older replay-source discrepancy. The leading unresolved explanations are conditional member utility, H512-specific native precision/scale effects, and unequal unknown TL3 exposure. Separate NumPy covariance/time-bin analysis and an eight-component native FP32/int8 probe are being prepared in isolated worktrees; neither is a coefficient search or a new scientific fit. Root owns bounded execution and interpretation.

I1 F0/B completed all 4,620 updates and the full-fold export in 2,053.26 seconds; official WP0.65907416 (t0 0.64821999, t1 0.66992832), finite coverage1.0/skips0. This is the 10%-component small-screen control, not a comparison with full-data TL3. F0/A now runs. CPU checks passed after removing a stale error-text regex in a pre-existing rejection test; see [verification receipt](QUALITY_RECEIPT_R2.json). No numerical/model change was needed.

## Saved-prediction geometry measured; precision probe running

The [R1 streamed diagnosis](endpoint_diagnosis/RESULT_R1.json) reproduces all three native scores within1e-15. H512 SWA improves its standalone point score by0.00022538 but changes the native endpoint by−0.00000553. The CP5 replacement increases target-weighted clipped-prediction SD by about1.07%/1.48%; label covariance changes−0.185%/+0.613%, so the variance cost exceeds the covariance benefit. Weighted H512 partial correlations with the labels conditional on TL3 are−0.1615/−0.1249, descriptive on exposedF0 only. Every fixed time bin loses; the earliest point delta is−0.01698 versus−0.00593 late. Independent source/math review passes and forbids causal or unseen-transfer inference.

A conditional interval extension failed on sequences with no scored rows in an early bin; the original result is preserved, and the adapter now leaves their zero moments intact. Rerun is pending the shared CPU slot. The [native precision probe](native_precision_probe/README.md) is active as `wnn-h512-native-precision-r1-20261021.service`/invocation`fece36127fdd492c9289736e2824d66c`, CPU1/RAM1.2GB/no-swap/600sec, zeroGPU/newfits. Its fixed eight-component selection hash is`eacc150ce8f30bf79e749678bdd1a79d9f61c77218aaa8175fefb8224a51d184`. Observed memory~0.43GB; i1 OOM counters remain0. Do not start another CPU-heavy job concurrently.

## Fixed-bin uncertainty complete; precision gate preserved

[R3 diagnostic](endpoint_diagnosis/RESULT_R3.json) completed35.96seconds under1CPU/1.4GB; format/check and independent dense-moment oracle passed. CP5 loses in every bin; earliest−0.016978conditional95%CI[−0.027560,−0.007647], latest−0.005931[−0.008317,−0.004246]. Early hasonly17,198of808,863scoredrows; the gap is not confined to initialization. F0exposure/unequalancestry remain unchanged.

[Precision execution](NATIVE_PRECISION_EXECUTION_R1.json) used383.82CPU seconds/0GPU/~0.504GBpeak. Feature-only outputs saved, CP5reconstruction passes2.1e-7, originalTL3fails0.002090versus1e-5. Targets were not accessed. H512nativeFP32 matchesTorch1.10e-6RMS; currentint8differs0.002655RMS. Root owns correction after dynamics_builder source-only diagnosis; no tolerance relaxation or scored claim. I1healthywithOOM0, F0/A finalepochbatch100lastobserved. FreshquotaR2still2,345.378seconds; recovery_audit now checks export-only B16 feasibility and repository reservations, no provider mutation.

## First matched i1 cell completed

[Interim F0/1001](I1_FIRST_MATCHED_CELL_R1.json): B0.65907416 versusA0.64317272; A−B−0.01590144,t0−0.01442246,t1−0.01738042. Rootmetadata confirms sameinitializer,normalization,fitmembership/subset; both4620updates/skips0. Cnowruns; F2andcomponentintervals pending. Sixfitpanelunchanged.

ExactoriginalcallbackR2completed24.21CPU seconds but againfailsoldbaselineparity0.002089739. CP5reconstructionstillpasses. Preservebothfailedgatesandallarrays; splitconstructiondoesnotexplainmismatch. HistoricalbaselinewasDockerPython3.11.16/NumPy2.2.6/PyArrow19.0.1/glibc2.36; currenthostruntimeclosureunder review. No labelaccess, precisiongrade, or tolerance relaxation. Mergedpublication source staticchecks and boundedCPU suites passed66+13tests with2data-dependent skips; receiptintegrationpending.

## Runtime sensitivity demonstrated; host reference replay admitted

[ContainerR3](native_precision_probe/container_original_endpoint_prelabel_validation.json) exactlyreproducesoldTL3cache(max0.0) onthefixed8groups, sameZIP/native/backend. This version-matchedimage isnotprovenhistoricalimageID, andnodependencycauseisisolated. HostR2differs0.002089739; neitherfailedgaterewritten. [Onefreshfullhostreference](NATIVE_HOST_REFERENCE_AMENDMENT_R1.json) nowrunsunitwnn-tl3-host-reference-r1-20261021/invocation9db1568a85b74074abd019e81c47fe29 at1CPU/1.4GB/2000sec/0GPU, resetABArequested, same468F0order. Independenti1_recipeLuna/maxreviewagreeswithseparatereference/grade andexplicitruntime/coveragegates. Rootownscompletionandgrader; oldcacheandgradesfrozen.

[InterimA/Bgrade](i1/interim_f0_ab/RESULT_R1.json) passed: A−B−0.01590144CI[−0.019737,−0.011972]; fixed90B10A−B−0.00002308CI[−0.000309,+0.000266]. EverypositionbinfavorsB; thissupportsstandardi1signal, notyetexplicithistorybenefit. No mixedendpointpromotion.

## Complete first i1 cell; private export launch

[Three-arm F0 points](I1_F0_THREE_ARM_POINTS_R1.json): B0.65907416, A0.64317272, C0.65553705. C−B−0.00353711 (t0−0.00208828,t1−0.00498593); C uncertainty/fixed-mixture contribution pending. Standardi1 has useful signal here, while this explicit history representation currently hurts; neither proves absence of temporal information. F2/B is running; allsix fits/checkpoints unchanged.

Full current-host TL3 reference completed468groups/9.36Mrows, ABAresetexact,373800KiBmaxRSS,1587.60wallseconds/1445.19callbackseconds. Targets not read; gated grade pending. Historicalcache preserved.

[B16 cloud admission](B16_CLOUD_LAUNCH_ADMISSION_R1.json) used [quotaR5](KAGGLE_QUOTA_R5.json) (2,345.378 seconds unreserved) and sealed private input v1; the remote archive matched. Root patched the two-T4 concurrency risk with [one-CPU inherited affinity](B16_KERNEL_SERIAL_AFFINITY_PATCH_R1.json), independently source-reviewed, and launched the exact [kernel source](../kaggle_h512_b16_cloud_export_r1/launch_r1/README.md). Kernel v1 ended `ERROR`: only CP5 started on GPU0/T4 UUID `GPU-39e6412f-b0e4-e371-bac5-5a682b136cc2`; it timed out at 705.03 seconds against a 700-second worker cap, with peak RSS 566,816,768 bytes and observed GPU use 0. SWA was not started. [Terminal collection](B16_CLOUD_TERMINAL_COLLECTION_R1.json) preserves the 187-file inventory, full provider stdout/stderr, worker specs, input identity records and failure receipts at the physical-scratch location. It confirms no prediction arrays or score. Both raw Parquet hashes were deferred to worker preflight (34,855,770,139 bytes combined); preflight hashing is the leading source-based explanation, but the worker log is empty so the exact stalled phase is unknown. Fresh [quotaR6](KAGGLE_QUOTA_R6.json) after the run shows 1,635.575 seconds unreserved and no provider reservation. No cloud retry is planned. Root will attempt both saved exports locally after the active local GPU job drains, within the separate two-hour recovery allocation.

## History contributes to a fixed blend; runtime hypothesis closed

The [complete F0 A/B/C interim grade](i1/interim_f0_abc/RESULT_R1.json) passes exact source, initialization, normalization, complete-cache and memory preflight checks. C−B is −0.00353711, conditional component 95% interval [−0.00728292,−0.00006648]. The frozen 90%B/10%C blend is **+0.00035953**, interval [−0.00001187,+0.00070893], with t0 +0.00052218 and t1 +0.00019688. All four blend position-bin point deltas are positive. A weaker standalone model can contribute useful errors to an ensemble; this one exposed fold/seed cell does not establish repeatability or generalization. F2 is the existing decision-changing comparison; no coefficient search or additional fit is admitted.

[Current-host native grade](NATIVE_HOST_GRADE_R1.json) passed all gates and reproduces the previous three scores. Host TL3 0.70181753 differs from legacy TL3 by −0.00000059; CP5−host remains −0.00736292, interval [−0.00961758,−0.00557430]. The runtime difference is real at prediction level but does not explain the endpoint quality gap. The [grade admission](NATIVE_HOST_GRADE_ADMISSION_R1.json) and [log](native_host_grade_r1.log) preserve the separately gated result; original caches and grades remain immutable.

The projected mean passed source/checkpoint metadata gates but has not yet reached parity or produced a new checkpoint. R1 rejected a plan entry missing the teacher coefficient's `family` field. R2 found only two additional child cache annotations (identity scale1.0 and preclip-order flag) absent from the parent metadata. Both failures are operational, not scientific negatives; [R1/R2 admissions](WEIGHT_MEAN_PROJECTED_ADMISSION_R1.json) and [R2 repair contract](WEIGHT_MEAN_PROJECTED_ADMISSION_R2.json), logs and prior plan preserve the attempts. Root is checking the exact metadata equivalence before a further bounded execution; no equality tolerance is relaxed and no GPU work has occurred.

## Fixed projected mean derived without fitting

[Derivation](WEIGHT_MEAN_PROJECTED_DERIVATION_R1.json) passes exact first-two-output parity for allthree sources and bitwise normalization checks. One equal serving-projected mean checkpoint now exists, SHA256 `fc5e432e…26a55`; [execution](WEIGHT_MEAN_PROJECTED_EXECUTION_R1.json) used9.86CPU-wallseconds/753480KiBpeak/zero swaps, noGPU ornewfits. Original full-state block and failedmetadataattempts remain historical. Full-fold export and frozen WMSE/oldZ4 comparisons wait for the single localGPU worker; no quality claim.

## Auxiliary target numerical guard failed before any fitting

[CPU fixture](DYNAMICS_NUMERICAL_PROBE_R1.json) confirms history-dependent cancellation in globalFP32energy-prefix subtraction: an early large pulse erases real later0.01changes, causing9.21044log-unit target error forbothhorizons. [Execution](DYNAMICS_NUMERICAL_EXECUTION_R1.json) used5.74seconds/657476KiBpeak, noGPU/fits. This is an implementation counterexample, not a measured dataset quality result. Dynamics profile admission is held for a stableFP32 mathematical repair and independent tests/review; i1_recipe owns isolated code, recovery_audit independent math review, root verification/admission. Active i1 source/recipe unchanged.

## Auxiliary numerical repair verified

[Verification](DYNAMICS_NUMERICAL_VERIFICATION_R1.json) passes six direct-window/causality/head-removal tests plus six checker tests, Ruff and canonical structure checks. Independent recovery_audit source/math review passes. [Original fixture rerun](DYNAMICS_NUMERICAL_PROBE_R2.json) error is5.76e−8log units after repair. Mathematical targets unchanged; no auxiliary fit/profile yet. Root owns corrected source freezing and sustained full-cost profile after the current GPU panel and fixed saved-artifact exports.

## Local saved-artifact preparation complete

[CPU stageR2](B16_LOCAL_CPU_STAGE_REPORT_R2.json) completed correctedfixture, clean exact68ffsource checkout, source verification and input symlink preparation in14.29seconds. R1sequence-only launcher error is preserved; no step ran before repair. BothsavedF1exports are ready for a later singleGPU admission; no inference orfit started. Source/model/score plan unchanged, and further Kaggle launches remain closed.

## Second control complete; merged numerical repair checked

F2/seed1002 B completed all4,830updates, skipped0, full443-group export/score: officialWP0.684108214809019 (t0 0.6838011873878901,t1 0.6844152422301477). A now runs; C remains pending. This is not a paired treatment result, and the absolute F0/F2 scores cannot identify fold versus seed effects. [BudgetR4](BUDGET_PROGRESS_R4.json) records four completed fits/exports/scores plus earlier profiles/failures, with active work additional.

[Publication-source verificationR2](MERGED_SOURCE_VERIFICATION_R2.json) passes merged corrected dynamics source Ruff, 63-file structure check, six checker tests and six auxiliary tests. The publication merge preserved all research-team README entries and retained its generated snapshot rather than replacing it with root-only generated navigation; merged tree freshness is not claimed. Corrected dynamics source is being frozen for a later profile, not admitted to GPU yet.

Correction to the cloud terminal paragraph above: the combined raw Parquet size is34,855,728,139bytes, as recorded in the exact input manifests; the earlier narrative total was a transcription error. No identity check or diagnosis changes.

## Native precision effect is small on the fixed diagnostic sample

[Fresh-host precision grade](NATIVE_HOST_PRECISION_PROBE_R2.json) passes both unchanged1e-5 reconstruction gates (original1.55e-7, CP5 2.10e-7) and actual feature-only need masks before targets. On eight preselected distinct exposed components/10,988scoredrows, TL3WP0.77775359 versusCP5 0.77443044. The frozenFP32counterfactual gains only+0.00004035 (t0+0.00003550,t1+0.00004520), leaving−0.00328280 versusTL3. FP32fast/exact agree to7.8e-10WP. Precision is measurable but doesnotexplainthissamplegap; nofullfold/publicextrapolation ornewpackagepromotion. Retainedpair/member scores arepredeclared diagnostics, notablationselection.

[Execution](NATIVE_HOST_PRECISION_EXECUTION_R2.json) used5.39wallseconds/382380KiBpeak/zeroSwap/GPU/fits. FirstlauncherR2failedbeforearrays/targets/outputbecause rootutils wasabsentfromPYTHONPATH; [corrected import admissionR3](NATIVE_HOST_PRECISION_ADMISSION_R3.json) preserves thesamesource/gates/outputpath andearlierfailedlog. No oldreceiptwasrewritten.

## Fixed history blend gain is not created by clipping

[Posthoc F0 geometry](i1/fixed_mix_geometry_r1/RESULT_R1.json) reproduces official scores to 2.22e-16, with covariance identity error below 1.64e-12 and variance error below 5.74e-13. The frozen 90%B/10%C mixture gains +0.00036506 in raw weighted correlation and +0.00035953 with official clipping. Clipping reduces the gain by 0.00000553. Overall covariance of the correction with the target after accounting for B is positive for both targets (0.000713004 / 0.000338789).

This supports a complementary prediction direction over pure scalar shrinkage or clipping-only explanations on these exposed fits. It does not distinguish useful history information from a changed optimization path, establish repeatability, or add independent evidence to the existing interval. Early t1 bins lose; the two later t1 bins improve. No coefficients were searched and no new models were fitted. F2 remains the next decision-changing comparison.

[Execution](I1_FIXED_MIX_GEOMETRY_EXECUTION_R1.json) used 20.04 CPU wall seconds, peaked at 441052 KiB, and used no swap or GPU. Independent dynamics_builder source/math review and root inspection passed before execution.

## Cold history treatment also changes the initial function

Root source inspection and independent dynamics_builder review confirm that C activates 156 random input columns which B masks, despite identical stored initial parameter hashes. This supplies an optimization explanation consistent with a weaker standalone C and a useful corrective direction; it is not a measured conditioning effect or proof of history value. The [interpretation review](../../../unified/reviews/I1_HISTORY_FIXED_MIX_GEOMETRY_REVIEW_20261022_R1.md#source-based-follow-up-interpretation) records the falsifier in the existing conditional warm-parent panel: zero appended weights, shared explicit new-slot normalization, exact parent-output parity, then only history off/on. No new panel, warm implementation or fit admission; F2 remains pending.
