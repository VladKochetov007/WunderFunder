# Frozen-prediction audit artifacts

This folder contains a no-fit audit of three existing validation prediction arrays. Start with [REPORT.md](REPORT.md) for the scope, results, limitations, and interpretation. The scoring used fixed cached outputs only; it did not fit a model, run new inference, or change a prediction cache.

## Results and interpretation

- [RESULTS.json](RESULTS.json) is the preserved canonical grading output. It contains the 14 fixed scenarios, per-target pooled metric moments, region and age summaries, bootstrap output, and input hashes.
- [RESULTS_CORRECTED_VIEW.json](RESULTS_CORRECTED_VIEW.json) clarifies presentation fields without changing scores. [BOOTSTRAP_SUMMARY.json](BOOTSTRAP_SUMMARY.json) separates observed fixed-panel deltas from bootstrap-distribution means. [REPORTING_CORRECTIONS.json](REPORTING_CORRECTIONS.json) records those field semantics.
- [FOLLOWUP_DIAGNOSTICS.json](FOLLOWUP_DIAGNOSTICS.json) contains cross-array correlations and 240 age × retrospective market-class × target records. Interpret its `weighted_target_variance_sum` using [AGE_CLASS_VARIANCE_CLARIFICATION.json](AGE_CLASS_VARIANCE_CLARIFICATION.json): this is a sum of within-sequence centered variances and omits between-sequence variation in target means. The weighted MSEs and correlations are unaffected.
- [ORACLE_DECOMPOSITION_CHECKS.json](ORACLE_DECOMPOSITION_CHECKS.json) independently checks the noncausal label-run oracle variance/covariance identity from directly scored prediction-variance sums.
- [SEQUENCE_SUMS.npz](SEQUENCE_SUMS.npz) stores additive per-sequence moments by scenario and target for independent sequence-level decompositions.

## Inputs, integrity, and provenance

- [INVENTORY.json](INVENTORY.json) records input prediction, checkpoint, target-cache, fold-cache, and parquet identities, finite coverage, and the fixed 467-row-group intersection.
- [ALIGNMENT.json](ALIGNMENT.json) verifies the target cache and official mask against every row of the original validation parquet in the shared groups. All recorded mismatch counts are zero. Historical prediction-array row order remains unverified because producer receipts are absent.
- [ARTIFACT_MANIFEST.json](ARTIFACT_MANIFEST.json) lists local artifact hashes, bound remote hashes, input identities, command provenance, and resource summaries. Its own hash is intentionally excluded to avoid a recursive value.
- [SOURCE_PATCH.json](SOURCE_PATCH.json), [GRADE_initial_failure.log](GRADE_initial_failure.log), [GRADE_retry.log](GRADE_retry.log), and [GRADE.log](GRADE.log) preserve the first report-assembly failure and the deterministic retry. The retry reproduced the saved sufficient statistics exactly; metric math was not changed.
- [INVENTORY.log](INVENTORY.log), [ALIGNMENT.log](ALIGNMENT.log), and [FOLLOWUP.log](FOLLOWUP.log) retain compact remote run summaries.

## Reproduction code

- [run_prediction_audit.py](run_prediction_audit.py) runs the fixed inventory and grading phases.
- [prediction_audit_core.py](prediction_audit_core.py) contains the predeclared output-only policies.
- [gmetric_standalone.py](gmetric_standalone.py) vendors the canonical centered pooled scorer used for this audit.
- [validate_cache_alignment.py](validate_cache_alignment.py) checks row-level target and scoring-mask agreement with parquet.
- [followup_diagnostics.py](followup_diagnostics.py) computes the cross-array and age/class diagnostics from existing caches.
- [correct_results_presentation.py](correct_results_presentation.py) creates the corrected presentation view; [summarize_bootstrap.py](summarize_bootstrap.py) creates the point-estimate/interval summary; [check_oracle_decomposition.py](check_oracle_decomposition.py) performs the independent oracle identity check.
- [test_prediction_audit_core.py](test_prediction_audit_core.py) has seven tests for the fixed transform functions. The audit folder passed those tests, Ruff check/format, and Python compilation.

The conditional validation slice has 467 shared complete row groups, all fold 0, and 846,654 official selected rows. The 512-chain bootstrap is conditional on these cached predictions. These exposed predictions are not claimed as pristine OOF or hidden-test estimates.
