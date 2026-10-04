# Final combo: i16 numerical verification passed; runtime pending

The full-data combo and fixed head are fitted. The fixed selection is parent
`c20261004_combo_s3_final_r1@ep3`, head
`c20261004_combo_head_s3_final@ep0`, equal output weights. Training used 10,606
eligible TRAIN sequences and 1,505 development VALID sequences, excluding the
368 historical holdout sequences from fitting. The historical holdout is not a new,
component-disjoint confirmation panel and did not choose the checkpoint.

The candidate archive is `submissions/c20261004_combo_s3_q16x/submission.zip`,
SHA256 `f61577da8745cfc545e9acc528e001198074729e3f0878656194ea93a0fb5ec9`,
5,165,014 bytes. It is **not yet cleared for submission**.

Twenty complete streams passed callback shape/finite/warmup checks. The q16x
v2 and v3 outputs are bitwise identical, and repeated sequence resets reproduce
outputs exactly. Against the native float reference, however, maximum raw
absolute error is 0.000415802, above the predeclared 0.000300 threshold. The
official clipped WP difference is only +0.00000008745, below its separate
0.000005 threshold. Preserve the failed raw-error gate; do not silently relax it
because the aggregate metric looks good.

The FP16 GPU prediction exports also differ from native q16x: maximum raw
absolute difference 0.040495, 99.9th percentile 0.026117, WP difference
+0.00008661. This comparison changes both inference arithmetic and execution
path. It does not yet identify an engine bug or an AMP cause. Both paths need a
common FP32 reference before that distinction can be made.

Next diagnostics, fixed before examining the failing rows:

1. Locate raw and clipped discrepancies, preserving sequence/row/target identity.
2. Export parent and head using explicit FP32 GPU inference, with new cache names.
3. Compare their equal mixture with the native float package and both q16x builds.
4. If the native float path matches, retain q16x as an accuracy/runtime tradeoff
   requiring an explicit decision; also evaluate a float package as the unchanged
   predictor with a different execution precision. No model retuning on this
   verification panel.

Raw evidence: `checks/e412_package_verification.json`,
`checks/e412_package_{q16x_v3,q16x_v2,f32_v3}.npz`, the sealed specifications,
and `runner/logs/e412-package-verification-v1.log`. The final verifier returned
exit1 as required; this is a numerical failure, with no OOM or missing artifact.
No competition upload was made.

## Completed arithmetic-path localization and i16 alternative

Against independent CPU Torch, nativefloat differs by at most2.235e-6 (WP difference−2.158e-9); fast versus exact native activation differs by5.96e-7. The remaining CUDAFP32 difference (~0.000594) therefore concerns arithmetic/execution differences rather than a demonstrated native implementation error. TF32 has not yet been isolated. FP16 versus FP32 GPU export differs by0.04058; preserve inference precision in all comparisons.

The original q16x failure is retained. A separately packaged i16 alternative uses the same fitted weights: `submissions/c20261004_combo_s3_i16/submission.zip`, SHA256 `21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb`,5,165,013bytes. Its full20stream comparison against nativefloat gives maxabs0.0001039505 and WPdelta−1.58286e-7, passing the unchanged3e-4/5e-6 gates. Actual callback precision is asserted as i16; v2/v3 outputs and1→2→1 sequence reset/replay are bitwise equal. Evidence: `checks/e412_i16_package_verification_v2.json`.

The first i16 reset verifier inadvertently inherited q16x environment settings and correctly failed its comparison; that attempt is preserved. The corrected verifier explicitly sets and asserts i16. GPU-concurrent timing is diagnostic only. Clean CPU timing and deployment runtime assessment remain required before this alternative is cleared. No competition upload was made.
