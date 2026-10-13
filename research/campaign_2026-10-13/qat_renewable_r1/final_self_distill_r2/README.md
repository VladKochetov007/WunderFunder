# Stationary final-parent distillation repair R2

The first final-parent QAT adaptations changed targets to the available F0 teacher cache and reduced the complete fixed blend diagnostic by 0.000502 (S3) and 0.001039 (S4). This explicit successor tests the original final CP5 initializer as a constant FP32 causal teacher, preserving inherited preprocessing and the matched 1440-update budget. All supplied evaluation groups were used by the original final parent; outcomes are fitted-data diagnostics, not held-out CV or public forecasts.

- [Frozen recipe](RECIPE_R2.json): source/checkpoint hashes, matched S3 FP32/W4 conditions, resource bounds and target change.
- [Component subset](fit_components_10pct.json): the same315 complete fitting components/1152 sequences, with successor experiment identity.
- [Profile sequence](PROFILE_SEQUENCE_R2.json): two sequential100-update actual-shape profiles with cache handoff and single-GPU admission.
- [Fit sequence](FIT_SEQUENCE_R2.json): two matched one-epoch fits; both completed 1,440 optimizer updates with zero AMP skips, as recorded in the terminal fit receipt below.

- [Profile sequence receipt](PROFILE_SEQUENCE_RECEIPT_R2.json) and [fit admission](PROFILE_ACCEPTANCE_FIT_ALLOCATION_R2.json): both100-updateprofiles finite, host<4GB andVRAM<2.18GB; root admits matched1440updatefits afterindependent sourceGO.

- [Fit sequence receipt](FIT_SEQUENCE_RECEIPT_R2.json) and [fit completion](FIT_COMPLETION_R2.json): both matched S3 continuations completed 1,440 optimizer updates/72 batches with zero AMP skips and identical recorded sampler RNG/normalization.

- [R2 QAT release specification](QAT_RELEASE_SPEC_R2.json) and [fixed recipe](SUBMISSION_RECIPE_R2.json): stationary original-teacher S3 learned-W4 archive replaces index3 only; three unchanged team members retained at0.25 each. Internal ZIP13,461,304bytes SHAe2f78e09b0147b37c74b5809b9aa38f48d1ee6b83d656483b14eeb609aac22b3 at `artifacts/qat_renewable_r1/submission_candidates_r2/self_s3/`, per-member callbacks are complete; fixed-blend grading and selection remain pending. [ABI receipt](ZIP_ABI_R2.json) passes all three identical portable ABI3 libraries, maxGLIBC2.34 and no libpython. [Build preflight failure](BUILD_PREFLIGHT_FAILURE_R2.json) records an absent physical parent, corrected before any package output existed.

- [Native export admission](NATIVE_EXPORT_ADMISSION_R2.json) freezes the three stationary CP5-initialized S3 vectors and their exact weights, archive and input hashes. [CPU admission correction](NATIVE_EXPORT_ADMISSION_R2_CPU_REPAIR.json) preserves one no-worker CPU-cap refusal and the root-authorized low-CPU re-admission. [Terminal export receipt](NATIVE_EXPORT_COMPLETION_R2.json) records all three complete 468-group native evaluations, exact prediction hashes, reset replay and resource peaks. Scores remain fitted-data diagnostics; independent fixed-blend grading is pending.

- [Original ZIP/native baseline parity](BASELINE_ZIP_NATIVE_PARITY_V3_R2.json): Python3.11 clean import, eightcomplete20k-rowF0sequences plusA→B→A; all179109requiredoutputs bitwise match the reconstructed fullpanel baseline, zero maxerror; links historicalnativebinary arithmetic to currentnativegrade. No score/runtime claim.

- [Selected ZIP v2 parity](SELECTED_ZIP_NATIVE_PARITY_V2_R2.json) and [v3 parity](SELECTED_ZIP_NATIVE_PARITY_V3_R2.json): actualPython3.11 ZIPcallbacks match independentR4 nativeblend bitwise on eightfullsequences+A→B→A,179109requiredrows each, finitef32 outputs/Nonewarmups. Pathsuse the identicalportablebinary; this is localdirectory/loader consistency, not an independenthardwareISA test.
- [Selected callback timing](SELECTED_CALLBACK_RUNTIME_R2.json) and [matched runtime comparison](MATCHED_CALLBACK_RUNTIME_R2.json): Python3.11 CPU0, same2complete20ksequences/tworepeats,87.564 versus89.678us/callback (ratio0.9764). Shortsequential localprofile only; filecachecap saturation/zeroOOM, unchangedresidentbyteweights, noINT4/officialhost speedclaim.

- [Selected QAT result and release](RESULT_AND_RELEASE_R2.md) and [hash-bound release receipt](RELEASE_R2.json): one13.46MB public-test ZIP; strongest testedQAT, near-original fitteddiagnostic, floatingcontrolhigher, Python3.11/fullsequence/ABI/runtimechecks pass, publicupload pendinganonymousportalaccess. Alllimitationsrecorded.

- [Independent selected-ZIP audit](../../../unified/reviews/QAT_FINAL_SELECTED_ZIP_RELEASE_AUDIT_R1.md): confirms publicfile/internalbuild hardlink and exactSHA/12safeCRC/hashmembers, source/ABI and selectedS3parity scope; GO for byteintegrity/offline checks only, no publicqualityclaim.

- [Navigation check](NAVIGATION_CHECK_R2.json):638local links acrossrootindex,topicREADMEs,QATreviews andrelease/deliverynotes resolve; review-path typos corrected beforedelivery. This is localnavigation evidence, not scientific/catalog freshness.

- [Generated catalog freshness](CATALOG_FRESHNESS_R2.json): treebuild succeeded and scientificstructure/ledger checks reachedthefreshnessgate, butcatalog/integrity snapshots became stale as nine existingbackgroundmonitorfiles changed. This is an operationalsnapshot limit; noclaimof fullresearch_treecheckpass.

- [Public transfer review](PUBLIC_TRANSFER_REVIEW_20261014_R1.md) and [exact user-reported receipt](PUBLIC_RESULT_20261014_R1.json): portal00fpe72k scored0.6680,54m26s/655MB;−0.0001 versus original,4m10s longer,5m34s deadline headroom. Archive17.46% smaller, no demonstrated public quality/runtime/RAM gain; exactprobecompleted, notpromoted. Portal timestamps/payloadidentity remain qualified.
