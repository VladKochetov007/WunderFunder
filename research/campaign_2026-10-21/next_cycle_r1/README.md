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

- [Serving-projected WMSE trajectory mean amendment](weight_mean_projected_r1/README.md): source review found that the pinned predictor and both native exporters use only the first two `head.2` rows, so a fixed equal mean of parent CP5 and WMSE CP0/CP1 becomes shape-compatible after the canonical served-output projection. The historical full-state mean remains blocked. A source-pinned derivation script requires exact checkpoint, membership, cache, normalization and layout checks plus pre-mean CPU FP32 projection parity; it is prepared only, with no checkpoint load, fit, prediction, or grade.

- [I1 profile admission/recovery](I1_PROFILE_ADMISSION_R2.json), [source hashes](I1_PROFILE_SOURCE_R1.json), [R1 plan](I1_PROFILE_SEQUENCE_R1.json) and [failure report](I1_PROFILE_REPORT_R1.json), [R2 plan](I1_PROFILE_SEQUENCE_R2.json): three synthetic invariant tests pass; original redundant width flags failed CLI validation before data, corrected to equivalent inherited widths. Sustained110batch profile admitted; no scientific fit. Raw logs `i1_profile_r1.log`, `i1_profile_r2.log` preserve attempts.

- [Budget ledger](BUDGET_R1.json): explicit stage ceilings, per-arm fits, reused controls and teachers; completed charges versus active profile reservations. [R2 profile report](I1_PROFILE_REPORT_R2.json) records a second CLI-only rejection: profile early exits cannot save training state. [R3 finite profile](I1_PROFILE_SEQUENCE_R3.json) removes that flag only for profiling; scientific fit recipes retain it.

- [Sustained profile result](I1_SUSTAINED_PROFILE_R3.json) and [terminal report](I1_PROFILE_REPORT_R3.json):1,100finite updates,4.36124s/batch,0OOM, GPUallocated0.98GB/reserved1.22GB;3GBcgroup had file-page reclaim. Provisional full-panel cost4.85h includes20%fitmargin/setup/export reserves; final command review still pending. Engineering evidence, not quality.

- [i1 six-fit packet](i1/README.md), [admission/source seal](I1_PANEL_ADMISSION_R1.json), [R2 sequence](I1_PANEL_SEQUENCE_R2.json), and [launcher repair](I1_PANEL_LAUNCH_REPAIR_R2.json): matched two-cell A/B/C implementation, eight passing CPU checks, sustained thermal profile, and live bounded execution. R1 had an omitted loader setting and startup OOM; preserved, charged, repaired explicitly. No quality result yet.
- [Dynamics review](DYNAMICS_REVIEW_R1.json): independent source audit passes the frozen recipe, with setup-cost and general-validator limitations; four CPU tests pass. [Canonical implementation/recipe](../../experiment_drafts/dynamics-aux-20261021/README.md) has no completed profile/fit.
- [B16 offline recovery](../kaggle_h512_b16_offline_recovery_r1/README.md): separate source-pinned saved-weight mean/export/grade adapter, preserving missing initialization digest and original failed acceptance. Independent review pending; zero new fits.

- [Memory-safety correction](MEMORY_SAFETY_CORRECTION_R1.json): user warning, exact R1kill diagnosis and correctedR2/native counters. Additional heavy admissions held while both active; host pressure remains a material constraint.

- [B16 independent review](B16_RECOVERY_REVIEW_R1.json): identified missing metadata byte binding, fixed with exact provenance/historySHA pins verified against terminalcollectionmanifest; reviewer confirmed resolution. Remaining source/mean/coverage/fixedgrade checks passed statically; execution pending.

- [One-off native source restoration](finish_native_source_restore.sh): waits for the exact native invocation to terminate, checks both old and committed exporterSHA hashes, and restores the committed dynamics exporter. CPU5%/RAM50MB/swap0, no model/data work. [Execution log](native_source_restore.log) records completion or refusal; shell syntax check passed.

- [First i1 epoch](I1_FIRST_EPOCH_R1.json): F0/B completed1,540updates in635.1seconds with finite gradients, zero skips and successful checkpoint/trainingstate save, OOM0. Execution evidence only; fixedCP2notyetcomplete.

- [Future-input dynamics finite packet](dynamics/README.md): four exact cold-fit/full-fold-export/score plans, sustained F0aux005 profile and fixed paired grader. Setup-cost accounting includes the full fitting-target normalization scan. Static checks pass; runtime/profile/fits pending.

[Draft PR #33](https://github.com/VladKochetov007/WunderFunder/pull/33) publishes the cycle implementation/recovery handoff against research-team. [Publication receipt](PUBLICATION_R1.json) records the isolated merge preserving both histories; no branch integration or quality promotion. Active fits keep their frozen source. Generated-tree refresh and merged-source checks remain pending a spare CPU slot.

- [Finite dynamics packet review](DYNAMICS_PACKET_REVIEW_R1.json): exact4fit/profile/consumer/pairedgrader wiring passed independent static review. Missing target-cachebyte binding was repaired in both screen graders before any screen quality result; reviewer confirmed the fix. Runtime verification remains pending.

## Completed native endpoint recovery

[Exact recovered grade](NATIVE_GRADE_R3.json) completes both468-sequence native replays and the fixed official comparison: originalTL3 **0.70181812**, H512CP5replacement **0.69445462** (delta−0.00736350), H512SWA345replacement **0.69444908** (delta−0.00736904). SWA−CP5 is−0.00000553, conditional95%component interval[−0.00029533,+0.00026122]. Both targets decline versusTL3. Replacement-versusTL3 intervals are[−0.00961812,−0.00557526] and[−0.00952880,−0.00564847]. These are exposedF0fitted diagnostics over141components; two retainedTL3members haveunknownancestry, so this is not cleanCV or a public-score forecast. The native source-restoration observer completed successfully. No newpackagepromotion.

## Investigation of the replacement gap

[The frozen diagnostic plan](ENDPOINT_DIAGNOSIS_PLAN_R1.json) asks whether native precision, prediction scale, or conditional ensemble utility explains why a reproduced standalone model loses as a fixed replacement. It reuses saved predictions, fixes position bins and covariance observables before reading new diagnostic results, and permits no blend or scale search. TL3 ancestry remains unknown. The investigation adds zero fits and zero GPU time. [Quality verification](QUALITY_RECEIPT_R2.json) records 67 passing compatibility tests plus a repaired stale error-message assertion and 13 passing changed-path regressions; canonical structure checks pass.

- [Recovered student exposure lineage](TL3_STUDENT_EXPOSURE_RECOVERY_R1.json) and [interim independent review](../../unified/reviews/H512_ENDPOINT_GAP_INTERIM_REVIEW_20261021_R1.md): an exact payload match links the replaced TL3 student to a final train+valid distillation recipe. This exposes an unequal F0 comparison; complete recursive ancestry is still missing. It does not explain away the public loss or admit another fit.
- [CPU verification plan](QUALITY_SEQUENCE_R1.json), [initial report](QUALITY_REPORT_R1.json), [initial log](quality_r1.log), [targeted follow-up log](quality_r2.log), and [verification receipt](QUALITY_RECEIPT_R2.json): Ruff and canonical structure checks passed. The compatibility batch passed 67 tests and found one stale error-message regex; that assertion was repaired, and its rerun plus 13 changed-path tests passed. No model, objective or fitting recipe changed.

- [Saved-prediction endpoint diagnosis](endpoint_diagnosis/README.md): streamed, source-bound analysis of covariance, scale, fixed time bins and differential native parity. All three native scores reproduce within1e-15; a dense-moment oracle passes. The conditional bin intervals are being added without changing endpoints or coefficients.
- [Eight-component native precision probe](native_precision_probe/README.md): fixed original/CP5 member reconstructions plus native FP32/int8 modes, selected before labels. [Admission](NATIVE_PRECISION_ADMISSION_R1.json) bounds the current CPU-only attempt; no new fit or submission candidate.

- [R3 fixed-bin diagnosis](endpoint_diagnosis/RESULT_R3.json): all four replacement losses remain negative under conditional component uncertainty; this is exposed endpoint evidence, not a clean architectural comparison.
- [Native precision execution](NATIVE_PRECISION_EXECUTION_R1.json): feature-only probe completed without OOM; original TL3 reconstruction gate failed before labels, so score interpretation remains blocked.
- [Fresh Kaggle quota](KAGGLE_QUOTA_R2.json) and [read log](quota_read_r2.log): read-only19:56UTCcheck again finds2,345.378free GPU seconds, providerreserved0, refreshOctober24; repository reservations and complete-task admission remain required.

- [B16 CPU derivation admission](B16_DERIVATION_ADMISSION_R1.json), [receipt](B16_DERIVATION_RECEIPT_R1.json) and [log](b16_derive_r1.log): verified saved CP3/4/5 and derived the frozen equal-parameter mean in a new scratch root. Zero newfits/GPUtime, initial tensor digest remains unavailable; exports/quality pending.

- [First i1 matched cell](I1_FIRST_MATCHED_CELL_R1.json): F0/1001 A−B−0.01590144, both targets negative when i1 is removed; identical initializer/normalization/membership/update counts. C/F2 and paired-component intervals pending; no promotion.
- [Exact-wrapper correction admission](NATIVE_PRECISION_CORRECTION_ADMISSION_R2.json), [receipt](native_precision_probe/original_endpoint_prelabel_validation.json) and [log](native_precision_r2.log): exact originalcallback again differs fromoldcache by0.002090max; targets remain unopened. This rules down the split-topology explanation, and runtime/source closure is under review.

- [Interim F0 A/B paired grade](i1/interim_f0_ab/RESULT_R1.json), [admission](I1_INTERIM_GRADE_ADMISSION_R1.json) and [log](i1_interim_grade_r1.log): removal loss persists under141-component resampling; fixed90B10A mixture is approximately flat. C/F2 pending.
- [Version-matched container admission](NATIVE_PRECISION_CONTAINER_ADMISSION_R3.json), [receipt](native_precision_probe/container_original_endpoint_prelabel_validation.json) and [log](native_precision_container_r3.log): oldTL3cache reproduced bitwise onall8selected sequences, implicating runtime differences rather than input/order changes; individualdependencynotisolated, exacthistoricalimageIDunrecorded.
- [Host-reference amendment](NATIVE_HOST_REFERENCE_AMENDMENT_R1.json) and [live log](native_host_reference_r1.log): one fresh complete468-sequence original reference replay oncurrenthost, CPU1/1.4GB/2000sec/0GPU, preservesoldcache andold1e-5gate; evaluatedsame-runtime endpoint deltas pending.

- [Merged-source verification](MERGED_SOURCE_VERIFICATION_R1.json), [finite sequence](MERGED_QUALITY_SEQUENCE_R1.json), [report](MERGED_QUALITY_REPORT_R1.json) and [log](merged_quality_r1.log): publication source passes Ruff/63-file training structure and66+13boundedCPUtests;2data-dependent tests skip withoutrawdata. No measured parity claim; generatedtree refresh pending.

- [Budget progress snapshot](BUDGET_PROGRESS_R2.json): two ofsixfits complete;4,736.31seconds completedphase/profile charges, plusactiveCtime. The5hour stage ceiling and12hour conditionalproduction reserve remain unchanged.
