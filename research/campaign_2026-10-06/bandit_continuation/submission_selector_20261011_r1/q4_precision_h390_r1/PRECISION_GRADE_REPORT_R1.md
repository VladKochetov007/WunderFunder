# H390 pair precision comparison — R1

## Finding

Keep the corrected Q8 VNNI package as the precision choice. On the same ten
20,000-row groups, the corrected Q8 package scored 0.76337826 WP versus
0.76344916 for the preserved R1 package, which actually ran FP32. The observed
Q8-minus-FP32 difference is −0.00007090 WP. This is a precision-fidelity
diagnostic on the H390 full-data pair, not cross-validation or a generalization
claim.

The Q4-rounded variant scored 0.76157701 WP, −0.00180125 versus Q8 on that
panel. Its ZIP is 3,357,848 bytes, 5,198,402 bytes smaller than the 8,556,250
byte Q8 ZIP. This variant stores rowwise rounded values as float32 for the same
Q8 VNNI loader; the separate code/scale NPZ is a research sidecar. It is not a
native INT4 package or a speed measurement. The measured score loss does not
support replacing Q8 with this variant.

The rounding arm applied 15 symmetric levels `[-7,…,0,…,7]` to each output row
of `gru.weight_ih_l0`, `gru.weight_hh_l0`, `gru.weight_ih_l1`,
`gru.weight_hh_l1`, and `head.0.weight`, using `row_max_abs / 7` scales and
round-to-nearest ties-to-even. Other arrays were left unchanged. The package
contains the reconstructed float32 arrays, which then pass through the same
Q8 VNNI quantization and native code as its source package.

## Exact comparison

| Package/runtime | t0 WP | t1 WP | global WP | Difference vs Q8 |
|---|---:|---:|---:|---:|
| Preserved R1, actual FP32 | 0.75302313 | 0.77387520 | 0.76344916 | +0.00007090 |
| Corrected R2, actual Q8 VNNI | 0.75304489 | 0.77371163 | 0.76337826 | 0 |
| Q4-grid rounded weights, Q8 VNNI | 0.75181137 | 0.77134265 | 0.76157701 | −0.00180125 |

The R1 ZIP contains `precision.txt` with `q8vnni`, but its `solution.py` ignores
that file and calls `wnn_native.Predictor(weights, fast_activations=True)` with
no precision argument. Instantiating the exact R1 ZIP against its bundled v3
library confirmed member precisions `["f32", "f32"]`. This agrees with the
preserved precision-retraction receipt; the older R1 callback's Q8 label is
retracted. R2 uses the corrected runtime, whose startup receipt records two
members at `precision=q8vnni`.

Both packages contain byte-identical model weight payloads and the same two
0.5-coefficient members:

| Member | Source checkpoint | Checkpoint SHA-256 | NPZ SHA-256 |
|---|---|---|---|
| 0 | `vt_all_h390_ds_homogeneous_s3_r3@ep5` | `04675f8220ba949c1dfb6c5796a3f25a9b8f655bb48f83c188bf8f9bacd361e0` | `62b11349b4372beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6` |
| 1 | `vt_all_h390_ds_homogeneous_s4_r3@ep5` | `cefee5b51cae7b72247a26558444f526e915a9fb7ad3927abd6cc02e63018419` | `9f019d54a1472beda7abbd35f7e006d2c967237070cbd5318c7a2f15bc0d1ec6` |

The exact R1 precision correction is recorded in
`research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/monitor_20261011_1837_r1/PACKAGING_PRECISION_RETRACTION_R1.json`
(SHA-256 `8d4f3cae2f786fde26f2d437eb04ec6ee68493cda6aad70d4bd0a4d0b12c7485`).
The R1 ZIP and prediction panel are
`/home/vlad/development/wunder/wnn_kaggle_t3_f0_teacher_r2_20261011/submissions/10-11-exploit-now_r1/h390_full_data_pair_q8/{submission.zip,PREDICTIONS_PANEL_R1.npy}`;
the R2 counterparts are in
`/home/vlad/development/wunder/wnn_kaggle_t3_f0_teacher_r2_20261011/submissions/10-11-exploit-now_r2/h390_full_data_pair_q8/`.
R1's callback receipt SHA-256 is
`31339e7d962edc2becabd9a29e4183dafe1785d5bc727d53311e6597bfa589c7`. R2's
runtime binding receipt SHA-256 is
`0b21f49c68afcff429d5e5ade6b9aeef1c7f5e635e506c88d6bc8ee21069cc8d`, and its
callback receipt SHA-256 is
`489acfce54cbd0cd95c21a982f8d93b12185d7affd8b5626d08c24fc4917ce20`.

## Evaluation and limits

The scorer was `global_wp_clip2_v1`, using `need_prediction AND is_scored`,
clipping predictions and labels to ±2, and weighting by the absolute clipped
label. It selected 15,615 rows from the first ten physical rowgroups of
`datasets/valid.parquet`, for sequence IDs
`256678, 751005, 1200028, 1892371, 1990332, 2308749, 4490992, 6746551,
8348957, 9147005`. Only `t0`, `t1`, `is_scored`, `need_prediction`, and `seq_ix`
were read from those rowgroups. Target and official-mask hashes match the
Q4 grade receipt. The 5,615,509,128-byte parquet was not fully hashed; its
physical rowgroup IDs, sequences, and per-group target/mask digests are bound in
the receipts. The feature-only prediction panel is SHA-256
`e5e6472cedfcb20ecf01195777ab986b3560379c5e81caea71973632dde0d6a3`.

The Q8-vs-FP32 prediction Pearson correlations over selected rows were
0.99999586 for t0 and 0.99999518 for t1; MAE was 0.00178279 and 0.00221543.
These are deterministic panel differences, not uncertainty intervals. No
all-data candidate is ranked here from its valid score.

The successful streamed grade ran through `scripts/limited` with one CPU,
1.2 GB process memory maximum, swap disabled, and a 600-second runtime ceiling.
The receipt reports 1,158,888 KiB maximum RSS. It reused saved R1/R2 predictions
and did not replay 200,000 model callbacks or read `valid_targets.npz`.

An earlier full-read metric attempt is preserved separately at
`/home/vlad/development/wunder/wnn_kaggle_t3_f0_teacher_r2_20261011/submissions/10-11-exploit-now_r2/h390_full_data_pair_q8/Q8_FP32_OFFICIAL_METRIC_ATTEMPT_R1.json`
(SHA-256
`ea43602eb556c567421778d53b7588f478a885d7f46c3486de6941334be0f9ac`). The
controller confirmed a kernel OOM kill at 2026-10-11 19:54:18 UTC: Python PID
1711892 in `run-p1711892-i5881629.scope`, with 1,169,008 KiB anonymous RSS,
64,472 KiB file RSS, and no swap. That attempt is a resource failure, not a
negative model result. The streamed successor is the scored evidence.

## Frozen evidence

| Artifact | SHA-256 |
|---|---|
| Final FP32-vs-Q8 receipt `artifacts_r1/grade_r3/FP32_Q8_GRADE_RECEIPT_R1.json` | `83de990426407dcd2ca92ee917c9dd7dce4b888b1b58da57b3dc01d2ec57d62f` |
| Earlier successful FP32-vs-Q8 receipt `artifacts_r1/grade_r2/FP32_Q8_GRADE_RECEIPT_R1.json` | `3ef3c425176829d591eb930143380131e13f2cd39e40ac679384411e7fc7517c` |
| Q4 official-mask grade receipt `artifacts_r1/grade_r1/Q4_GRADE_RECEIPT_R1.json` | `2751abfc7a68c33895be0a32d2da37bbc229dbd4de61c4bc9c71330f79a8f479` |
| Q4 build receipt `artifacts_r1/Q4_BUILD_RECEIPT_R1.json` | `f6a04c25ad9548833b65ad322ccee06a4b158759d9f068ba52a28de0b21e6154` |
| Q4-grid package ZIP | `d3a6a80a69e3c3ab64b3ed23d99a2367ee32532c87dda5da1e9032fb0e959f58` |
| Q4 code/scale sidecar | `870e9afe5cc0357838cf3051fb05169335b7bfa6c31d0facf35c78ad45da1c68` |
| Q4 saved predictions | `f7ecba6a32a5e5cc7c66105f67f2cff3a41e91af9c4e395735aec459bd2b997d` |
| R1 FP32 ZIP / predictions | `6ff0d6c98a8cbec8ccd26866a74c39c69c2f6f07f1d5850735eb24634d09bae5` / `31a8b2d426a516ff0df107e12a39d237a90da2cd806c18262c814b3140f11dc1` |
| R2 Q8 ZIP / predictions | `3420ea5e6318e15b6340582125959ed7489983b17338f399c748965cb42c4a4a` / `682376c1f056efcd4a011d5e5edd5b3de585f1337e5f93313c3f1bf89a2e04c5` |
| Bundled native v3 library, both packages | `ce179a68b7cd321368cc9ab73cdf1b79973afcdb8266ef8f6d7e613bb6f9874e` |
| R1 precision-retraction receipt | `8d4f3cae2f786fde26f2d437eb04ec6ee68493cda6aad70d4bd0a4d0b12c7485` |
| Metric implementation `src/wnn/gmetric.py` | `2cf10e740a831379f62b7d1269456053f5c1f9efa2dbab03156e366972bed5aa` |

Implementation pins at grade time: `build_q4_variant.py`
`e85bf891d3f5c38f62c58201f1debd31c4437abea37f63d07d4da69852c02471`,
`grade_q4_variant.py`
`8823ff5a89b9be6e8f6d3367b663f83a2980144dece9f88029639f1d562e6c38`, and the
final streamed grader `grade_fp32_vs_q8_panel.py`
`f7ffa758735985c857017372c40136b1ce6c7b3cf5a1b709118755b653ebc97c`.

`ruff check` passed for this task directory; the focused rounding tests passed
5/5 under a one-CPU, 700 MB, 120-second limit.
