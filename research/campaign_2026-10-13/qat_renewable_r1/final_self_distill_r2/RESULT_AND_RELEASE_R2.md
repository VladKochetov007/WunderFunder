---
kind: result
status: inconclusive
created_utc: 2026-10-13T21:14:40.416283+00:00
updated_utc: 2026-10-13T21:14:40.416283+00:00
hypothesis_ids: [H-QAT-LSQ-20261013]
experiment_ids: [E-QAT-FINAL-SELF-DISTILL-20261013-R2]
evidence_paths: [RELEASE_R2.json, MATCHED_CALLBACK_RUNTIME_R2.json]
---

# Selected QAT public-transfer package

## Question or claim

The strongest tested QAT fixed blend is ready for the user's requested public test. Its 13,461,304-byte ZIP preserves exact learned scales and replaces only the fourth member of the original equal-weight blend. The available portal session is anonymous; no upload or public score exists for this file.

## Mechanism and method

Stationary original all-data H390 CP5 seed3 teacher, FP32 causal states, learned recurrent W4 weight scales and Q8 first head. Same315 complete fitting components/1152 sequences as the matched FP32 control, batch16/chunk1000,1440 updates, inherited normalization, fixed raw epoch0. No activation rounding in training; this is weight-only QAT. Three original team members and all four0.25 coefficients remain fixed.

## Evidence and interpretation

Independent full native diagnostic (468 sequences,808863 selected rows): baseline0.702266238; selectedQAT0.702216145; matchedfloatingQ8 control0.702374530; matchedW4PTQ0.702132829. QAT minus baseline−0.000050093, versus matchedW4PTQ+0.000083316, versus floatingQ8−0.000158385. QAT improves on old R1S3 by0.000451955. The floatingQ8 control has the highest point; the selected file is the strongest tested QAT transfer probe, not a claim of best overall quality.

Actual selected ZIP V2/V3 paths match the independently graded blend bitwise on8full20k-row sequences plusA→B→A,179109 required predictions perpath. Python3.11 cleanimports, Nonewarmups/finitef32 outputs, safe12members/CRC/source hashes and all-library ABI pass. The17-test archive/native suite used an actualS4 fixture; selectedS3 has direct fullsequence parity instead.

Matched local Intel i9-11900H CPU0/Python3.11 profile: selected87.564us percallback versus baseline89.678us, ratio0.9764,2sequences/tworepeats. Both hit the1GB filecache cap while checksuming fulldata, with zeroOOM/kill/throttle. Resident streamedweights remain unchanged. WholeZIP17.46% smaller; nativebinary changes also contribute, so this is not solely aQ4 packing saving.

## Limits and validity

Original final parents fit every supplied validation group. This panel is an exposed fitted-data diagnostic, not cleanCV or a publicscore forecast; no interval or independentfold confirmation accompanies the final comparison. The propensity-cache NPY payload hashes are not in standard run provenance. Q4 unpacks once into bytekernels, with no INT4 execution. Two sequential local runtime repeats do not establish official AMD-host speed or deadline guarantees. A different directory path is not an independent hardwareISA test.

## Decision and next step

Release one hash-bound QAT package for the explicitly requested public transfer test. The user can upload it through an authenticated portal session. Retain floating controls, earlier negatives and checkpoints. The wider plan continues to F3/allfold and learned-scale/activation-aware validation; archive compression alone does not justify claiming a leaderboard improvement.

## Source records

[Release receipt](RELEASE_R2.json), [frozen recipe](RECIPE_R2.json), [native completion](NATIVE_EXPORT_COMPLETION_R2.json), [independent grade](../../../unified/reviews/QAT_FINAL_SELF_DISTILL_BLEND_GRADE_R4.md), [V2 parity](SELECTED_ZIP_NATIVE_PARITY_V2_R2.json), [V3 parity](SELECTED_ZIP_NATIVE_PARITY_V3_R2.json), [ABI](ZIP_ABI_R2.json), [runtime comparison](MATCHED_CALLBACK_RUNTIME_R2.json), [delivery file and README](../../../../submissions/10-14-qat-r1/README.md).
