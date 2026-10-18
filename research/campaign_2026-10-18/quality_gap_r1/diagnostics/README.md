# F0 residual diagnostic

Status: independently reviewed, not staged or executed. SSH access to Beleriand currently fails before authentication; no model/array analysis has run from this packet.

[INPUTS.json](INPUTS.json) binds the existing compact original-label F0 rows, wide/deep predictions, frozen 75/25 FP32 mixture and aligned native Q8 output. All input files and extraction/blend/replay/grade receipts are hash-bound. This is a retrospective decomposition of an already inspected panel, not a comparison with the stronger incumbent, a new blend search, calibration fitting or public-score explanation.

[PREPARATION_MANIFEST.json](PREPARATION_MANIFEST.json) records exact code and input hashes, runtime paths, intended finite CPU allocation and output. [DIAGNOSTIC_REVIEW.json](DIAGNOSTIC_REVIEW.json) independently checks formulas, local receipt schemas, the staged import layout and three small synthetic tests. Review found a zero-weight-bin failure before execution: paired descriptive values could be undefined. The fix preserves `None` while keeping additive contributions zero; the regression test passes. Root also caught an earlier CSV extra-column mismatch and clarified that slopes regress prediction on target. No training code changed.

## Computation

The entrypoint [diagnose_f0_residuals.py](diagnose_f0_residuals.py) validates pinned inputs through [input_bindings.py](input_bindings.py) and [receipt_validation.py](receipt_validation.py). [residual_moments.py](residual_moments.py) computes globally centered official weighted-Pearson terms, and [residual_analysis.py](residual_analysis.py) assembles JSON/CSV tables. [Tests](test_residual_analysis.py) cover additive identities, paired differences, zero-weight bins and actual CSV writing.

For each target, each partition contributes one half of its globally centered weighted covariance divided by the global denominator. Contributions sum exactly to the official pooled WP; paired partition differences sum to the endpoint difference. Label-magnitude bins and components share membership across models. Prediction-magnitude and clipping bins are model-specific descriptive tables, so their rows are not paired. Within-bin correlations are descriptive and not additive. Prediction-on-target regression slopes are not coefficients of a fitted target-on-prediction calibrator.

Sequence positions are unavailable in the compact bundle; no raw-parquet scan is scheduled. The strong-control comparison requires separate source-qualified exports. No diagnostic output yet establishes where that control gains or explains public transfer.

## Remote execution after access returns

Preserve repository-relative layout beneath the campaign root, including the four exact `f0_submission_r1/receipts/` JSON files referenced by INPUTS. Existing remote paths must match hashes or fail closed; never overwrite historical receipts. Set `PYTHONPATH=B/source-a266b036/src`; the entrypoint's inferred `B/src` is absent and canonical imports must resolve through this environment. Hash the actual imported gmetric and record it with the execution receipt.

Rehash the prepared files and check fresh aggregate headroom. Execute one support unit named `computer-vision-homework-quality-gap-diagnostic-v1` inside `wnn-campaign.slice`: 50% CPU, 1.3GB high/1.5GB max, swap0, 300s, CUDA hidden, one BLAS/OMP thread and physical `B/research_tmp`. Do not overlap another support task. Invoke the entrypoint with `--inputs INPUTS.json --output B/quality-gap-r1/diagnostics-v1/RESULT.json`. Paths and outputs are create-only. Root owns admission/execution; preparation is not a launch.

After exit, capture actual status, runtime/resource information, both result files and source hashes as exact bytes. Check the internal score/contribution assertions and inspect results before interpretation. Capture then stop only the own exited support unit. A partial JSON without CSV or nonzero exit is not accepted. Do not duplicate existing outputs after interruption.
