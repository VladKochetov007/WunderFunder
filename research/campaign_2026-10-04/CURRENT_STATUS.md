# Continuation status

Campaign experiment tally: **14/17 Oct4 continuation experiment records complete; 2 partial (E-414 calibration only, E-419 local result but no QAT/PTQ fit); 1 queued (E-407 TPU).**

As of 2026-10-04T11:00:39.437879+00:00

- Best public score remains0.6641. Full-data combo i16 package passed full-stream numerical/reset checks and clean local runtime. It is ready for user upload; no new public score exists. Public-reference-anchored runtime projection40.3–43.5min is uncertain.
- E410 complete combo predictor gain is+0.002414, interval[+0.001566,+0.003449], on exposed folds. The package has not been submitted.
- E413 volume-history pair: one exposed fold/seed+0.0009618, interval[−0.0000800,+0.0020597]; needs replication.
- E419 local FP16/FP32 paired result is near zero: FP32−FP16−0.0000160, interval[−0.0001502,+0.0000864]. Kaggle v3 used both T4s successfully for preflight, production-shaped probe, a matched FP32 control and export; stale prediction names caused the score step to fail. No PTQ or QAT fit ran. Recovered v3 FP32 controls pass corrected local single-arm scoring (WP0.681496); both copies have identical weights/predictions, so this validates the score path only. Four GPU-hours charged;26 GPU-hours and20 TPU-hours remain. Exact failure evidence and artifacts are preserved under kaggle/E419_qat_dualT4_v3/.
- E420 native factor parity and timing complete. Factor backend458.20kcycles versus effective dense FP32 763.03k and original q16x W384 396.07k; no speedup over the existing quantized engine.
- E422 reciprocal mature-gain screens: hidden-state and input-summary gain R² both negative; earlier-label multiplier reduced later-episode WP. This narrow probe does not establish representation exhaustion.
- E423 EMA-in-rows vs fresh same-source control:−0.000574 WP, interval[−0.001797,+0.000303],468 sequences/141 components. One exposed fold/seed; no gain signal.
- H093/H094 specialist and prefix-adapter implementation merged; production objectives/teachers/controls corrected.35 focused CPU tests passed; production model fits remain pending.
- TPU remains QUEUED, no device probe. Width fold1 pair staged but not launched. Local quota/security guards remain active; no OOM in completed E423 runs. The earlier E422 CPU-helper OOM remains preserved.
- Further scientific candidates remain pending: fresh-view training, matched normalization and credit experiments, specialist/adapter model fits, width replication, QAT retry and structural teacher. Latest audit checked all7 Kaggle kernels:3 COMPLETE,3 ERROR,1 QUEUED,0 RUNNING. E413 is only completed scientific Kaggle result. GPU quota26/30h, TPU20/20h; refresh Oct10. Full table: KAGGLE_KERNEL_STATUS_2026-10-04.md. Luna-only delegation, maximum six.

Reports are in E412_READY_REVIEW.md, E419_LOCAL_PRECISION_REPORT.md, kaggle/E419_qat_dualT4_v3/failure_report_20261004T1107Z.md, E420_factorized/RUNTIME_REPORT.md, E422_GAIN_REPORT.md and E423_EMA_REPORT.md. Raw results, plots and resource attempts remain in checks/, eda/, and runner/.
