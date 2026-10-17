# R2 frozen-prediction audit

This is a no-fit audit of three existing validation prediction arrays. It applies a fixed, predeclared set of causal output transforms and one explicitly noncausal label-run oracle, then scores them with `global_wp_clip2_v1`. No model inference, optimizer step, calibration, coefficient fitting, or prediction-array mutation was performed during this audit.

## Coverage and identity

The candidate arrays were selected by lineage before scoring: two `t384` seeds and one distinct `tiedlead` family. Each is shaped `(1873, 20000, 2)` and stored as `float32`. The `t384` arrays are finite on all 1,873 row groups; the `tiedlead` cache is finite on 467. The analysis uses their whole-row-group intersection: 467 row groups, all fold 0, and 846,654 official selected rows. It does not discard individual rows from an otherwise incomplete group.

[`ALIGNMENT.json`](ALIGNMENT.json) verifies all 9,340,000 rows in the common groups against `valid.parquet`: sequence IDs, step indices 0–19,999, both target columns, `is_scored`, the `need_prediction == (step_in_seq >= 99)` rule, and the final official mask all match exactly. Every mismatch count is zero. The three inspected checkpoints explicitly contain empty `lookahead` settings.

The historical prediction arrays have no producer receipt. Their linkage to the inspected checkpoints, source version, and assumed row-group order therefore remains unverified. Exact target-cache alignment does not establish prediction-array order. Treat all scores as conditional on those arrays already being in the stated cache row order. These exposed validation predictions are not claimed as pristine OOF or hidden-test estimates.

## Fixed diagnostics

The 14 scenarios are baseline; hold output until an `i0_p0 AND i0_p11` change; the corresponding OR control; fixed update cadences 2, 4, and 8; a 50% baseline/AND-hold blend; raw-output and clip-before-EMA variants at fixed `alpha=1/tau` for `tau=4,16,64`; and the clipped-label-run oracle. All causal transforms change only emitted predictions from the frozen arrays. They do not skip or replay backbone hidden-state updates. The raw-output EMA is clipped by the metric; the other EMA first clips each cached output and then applies the EMA.

The oracle averages already-clipped predictions over selected rows within each target-specific run of constant clipped labels. Its run boundaries use future labels and its means use multiple rows from each run, so it is not causal or deployable. [`ORACLE_DECOMPOSITION_CHECKS.json`](ORACLE_DECOMPOSITION_CHECKS.json) checks the variance partition against the oracle’s *directly scored* prediction variance. All six target/model partitions and covariance checks pass with relative errors below `7.6e-10`; the score identity errors are below `4.4e-10`. The raw `RESULTS.json` also contains a derived partition field; use the supplemental checks for the independent direct-variance validation.

The AND-event clock has 175,480 selected rows; the quiet complement has 671,174. Only 35 selected rows have any raw target change outside the AND clock. Holding emissions at events leaves those event-row outputs unchanged. For all three arrays, the held policy’s quiet-region WP drops substantially, while its 35-row exception-region score remains a tiny-sample diagnostic. The age buckets reset at the start of each sequence, so the oldest observed age is left-censored by the sequence boundary.

## Results

Canonical pooled WP on the shared fold-0 slice:

| Frozen array | Baseline WP | AND-hold delta (paired 95% chain interval) | OR-hold delta | Cadence-2 delta | Label-run oracle delta |
|---|---:|---:|---:|---:|---:|
| `pred_t384_s3_full_valid.npy` | 0.701304 | −0.117720 [−0.138135, −0.098412] | −0.088484 | −0.004458 | +0.029115 |
| `pred_t384_s4_full_valid.npy` | 0.702173 | −0.118358 [−0.138987, −0.099253] | −0.089271 | −0.004549 | +0.029180 |
| `pred_dftt_tlead_s3_valid.npy` | 0.706250 | −0.125650 [−0.148447, −0.105082] | −0.096824 | −0.004600 | +0.030070 |

Every prespecified causal output transform has a negative point delta on all three arrays. The fixed cadences and EMAs are negative controls for simple output smoothing/holding; the complete scores, per-target metrics, event/quiet/exception strata, age-bin weighted MSEs, exposure, and all conditional bootstrap intervals are in [`RESULTS.json`](RESULTS.json). The canonical point deltas and bootstrap-distribution means are separated in [`RESULTS_CORRECTED_VIEW.json`](RESULTS_CORRECTED_VIEW.json); the companion [`BOOTSTRAP_SUMMARY.json`](BOOTSTRAP_SUMMARY.json) provides a compact version.

The oracle raises WP by about 0.029–0.030 while removing about 7.6–8.1% of baseline prediction variance within constant-label runs. That gain comes from a future-defined transformation, not an online policy. It shows that output variation inside persistent-label runs matters to this score; it does not distinguish new information arrival from recurrent settling or another source of temporal variation.

The three arrays are strongly correlated with one another on this shared slice: pairwise weighted correlations are 0.9909–0.9925 for clipped predictions and 0.9902–0.9922 for clipped residuals. After subtracting each target’s mean within each constant clipped-label run, the remaining output fluctuations still correlate at 0.9604–0.9632. These related seeds/families therefore do not provide independent evidence that within-run variation is model-specific signal; the correlations also remain conditional on the unverified historical cache ordering.

For the representative `t384_s3` baseline, age-bin MSE rises from about 0.476 to 2.047 for t0 and 0.472 to 2.358 for t1 between the youngest and oldest bins, but age is confounded with market-class composition and the oldest bucket is left-censored. The follow-up reports 240 age × retrospective `market_class` × target strata across the three arrays (80 per array), including selected-row counts, absolute-clipped-target weight mass, a per-sequence centered weighted target-variance sum, and baseline/AND-hold weighted MSE. The variance field omits between-sequence variation in target means, so it is not the pooled variance of the whole age/class stratum; see [`AGE_CLASS_VARIANCE_CLARIFICATION.json`](AGE_CLASS_VARIANCE_CLARIFICATION.json). This does not affect the reported weighted MSEs, which use their weighted squared-error numerators divided by weight mass. Within-class patterns are heterogeneous; this descriptive breakdown does not establish that age itself causes worse predictions. `market_class` is target-derived and is used only retrospectively for stratification.

The 512-replicate paired chain bootstrap resamples validation chain clusters conditional on these fixed arrays. It is not an estimate of fit, seed, model-selection, or holdout uncertainty. The candidate seeds are related models, and all common groups are fold 0.

## Artifacts and reproducibility

- [`INVENTORY.json`](INVENTORY.json) contains candidate hashes, complete-group coverage, inspected checkpoint args, target/fold/data hashes, and current-source hashes. The full parquet SHA-256 is `2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782`.
- [`SEQUENCE_SUMS.npz`](SEQUENCE_SUMS.npz) stores six additive weighted moments (`w, wy, wp, wyy, wpp, wyp`) per sequence, scenario, and target for independent between/within-sequence decomposition. Its local and remote SHA-256 is `febb297f3c3edb873f8b5b4f82582249031332246599bd75fbeb952de9c56cde`.
- [`gmetric_standalone.py`](gmetric_standalone.py) is the copied canonical centered pooled scorer. [`prediction_audit_core.py`](prediction_audit_core.py) contains the fixed causal transforms and oracle. [`test_prediction_audit_core.py`](test_prediction_audit_core.py) exercises the real transform functions; all 7 tests pass.
- [`SOURCE_PATCH.json`](SOURCE_PATCH.json) preserves the first report-writer failure and records the one-line indexing correction. The retry reproduced the first pass’s saved sufficient statistics exactly; the score calculations were not changed. The original failed log remains [`GRADE_initial_failure.log`](GRADE_initial_failure.log).
- [`REPORTING_CORRECTIONS.json`](REPORTING_CORRECTIONS.json) documents presentation-only corrections without modifying raw `RESULTS.json`: the bootstrap `mean_delta` is renamed `bootstrap_mean_delta`, the observed fixed-panel delta is separately computed, exposure is explicitly labeled as changed rows, and the oracle’s partition is independently checked using direct variance sums. [`RESULTS_CORRECTED_VIEW.json`](RESULTS_CORRECTED_VIEW.json) and [`BOOTSTRAP_SUMMARY.json`](BOOTSTRAP_SUMMARY.json) provide the clarified view; all score, moment, replicate, and interval values remain unchanged.
- [`FOLLOWUP_DIAGNOSTICS.json`](FOLLOWUP_DIAGNOSTICS.json) and [`FOLLOWUP.log`](FOLLOWUP.log) contain the bounded cross-array and age-by-class pass (17.5 seconds, one CPU thread, 927,216 KiB maximum RSS). The follow-up read the frozen prediction arrays and checked that their hashes did not change.
- [`ARTIFACT_MANIFEST.json`](ARTIFACT_MANIFEST.json) lists local artifact hashes, remote source paths and hashes, and exact run commands. The full remote output directory was `/home/grinchenkoeg/wunder_research_reset_20261017_r2/prediction_audit/`.

The remote inventory, grade, and alignment runs capped OpenMP, MKL, OpenBLAS, and WNN threads at one. The canonical retry reported 358.5 seconds elapsed and 1,101,612 KiB maximum RSS. Local validation: `.venv/bin/pytest -q research/campaign_2026-10-17/research_reset_r2/prediction_audit/test_prediction_audit_core.py` (7 passed), Ruff, and `py_compile`.
