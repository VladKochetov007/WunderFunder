# Component-paired bootstrap for QAT deployment comparisons — 2026-10-13

**Status:** the bounded grader is implemented and its native-Q8 anchor has been checked against the actual frozen R2 full-fold artifact. The first graded pair compares two unchanged-parent deployment baselines (native Q8 versus recurrent W4); no trained-QAT-versus-matched-float-PTQ pair has been graded yet. The machine-readable [receipt](receipts/QAT_NATIVE_Q8_BASELINE_VALIDATION_R1.json) records the source hashes and paired baseline result; [the grader](../../../scripts/grade_qat_component_bootstrap.py) is ready to consume additional matched endpoint directories.

## Estimand and resampling unit

This consumer compares fixed, complete deployment predictions on the same exposed F0 evaluation panel. The paired estimand is candidate minus reference for `t0`, `t1`, and their official equal-weight mean (`wp`), using the frozen deployment evaluator's per-component six-sum arrays. The bootstrap resamples the 141 source-qualified connected components with replacement, drawing 141 component IDs per replicate. Each replicate uses the same sampled component multiplicities for reference and candidate, pools each endpoint's six raw sums, scores both targets with the pinned `wnn.gmetric.wp_from_sums`, and records the paired difference. It uses NumPy `default_rng(20261013)`, 10,000 replicates, batches of 128, and NumPy's linear quantile interpolation for the 2.5th, 50th, and 97.5th percentiles alongside the full-fold point difference.

## Identity and mask checks

The grader requires the admitted R2 manifest SHA-256 `8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`, the exact ordered 468-group F0 membership, and the expected mapping to 141 components. It verifies all source-file hashes named by the manifest and checks each endpoint's manifest identity, `profile_only=false`, exact full-sequence replay, score metric, group order, component IDs, and component-sum shape. Against the native-Q8 anchor it also requires the same selected-row total and exact per-component target-only sums (`W`, `WY`, `WYY`) for both targets. These statistics strongly check label/mask identity but are not a row-level mask digest; different masks with colliding per-component sufficient statistics are not ruled out mathematically.

Before grading a pair, the grader pools the anchor's component sums and checks `wp_from_sums` against the result receipt's stored `t0`, `t1`, and `wp` with absolute tolerance `1e-10`. This independently reproduced the native-Q8 point scores to within `1e-15`.

The first baseline-only pair uses recurrent W4 as reference and native Q8 as candidate. The full-fold WP difference is `+0.0008468`; its 10,000-replicate raw-sum bootstrap 95% interval is `[-0.0002493, +0.0022007]`. Target deltas are `+0.0013595` for t0 (interval `[-0.0001689, +0.0034892]`) and `+0.0003340` for t1 (interval `[-0.0006149, +0.0012818]`). These unchanged-parent quantization baselines are not trained QAT and the intervals are descriptive, not an automatic kill rule.

## Numerical limitation and interpretation

The producer archive exposes the six legacy raw sums. `gmetric.GlobalAccumulator.result()` scores with centered moments to preserve precision on nearly constant data, and those per-component centered moments are absent from the archive. The resampled intervals therefore use the official module's `wp_from_sums` function on the raw-sum aggregates; they are a descriptive raw-sum bootstrap approximation, not an exact bootstrap of the centered-moment scorer. The observed full-fold point-score agreement is excellent for the native-Q8 anchor but does not prove every resample has negligible cancellation. If exact centered-moment intervals are needed, a future frozen evaluator output must include per-component centered moments (or sufficient prediction/target data to reconstruct them); this grader does not change the evaluator.

The bootstrap conditions on fixed endpoint predictions. F0 has prior research exposure and is not pristine cross-validation. Intervals describe evaluation-component uncertainty only; they do not measure training-seed variation, establish public transfer, or act as an automatic stop/kill rule. The grader verifies prediction-artifact structure and score inputs, but it does not audit model ancestry, fitting membership, or recipe matching; those claims require separate provenance review. The unchanged-parent quantization baseline pair is an engineering comparison, not evidence about trained QAT.

## Invocation

Run from an environment with the repository's existing NumPy dependency. Use absolute paths when consuming artifacts outside the review checkout. Once both actual deployment endpoints exist, an example is:

```sh
python scripts/grade_qat_component_bootstrap.py \
  --repo-root /home/vlad/development/wunder/wnn_connectome_starterpack \
  --manifest /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/qat_renewable_r1/deployment_frozen_r2/manifest.json \
  --expected-manifest-sha256 8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556 \
  --baseline /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/qat_renewable_r1/deployment_frozen_r2/native_q8 \
  --pair native_qat_vs_float_ptq /absolute/path/to/native_qat /absolute/path/to/float_ptq \
  --output /absolute/path/to/new_grade_receipt.json
```

The output is create-only. Each pair endpoint must contain `result.json` and `component_sums.npz`, and either an adjacent `manifest.json` or the pinned baseline manifest identity. This method does not launch fits, alter the official evaluator, or promote any candidate.

Related experiment: `E-QAT-DEPLOYMENT-20261013-R1`. Baseline source: `artifacts/qat_renewable_r1/deployment_frozen_r2/native_q8`; deployment manifest: `artifacts/qat_renewable_r1/deployment_frozen_r2/manifest.json`.
