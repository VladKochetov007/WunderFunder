# Component-paired bootstrap for QAT deployment comparisons — 2026-10-13

**Status:** the bounded grader is implemented and its native-Q8 anchor has been checked against the actual frozen R2 full-fold artifact. It grades the unchanged-parent native-Q8 versus recurrent-W4 baseline and six trained pairs: uniform-W4 and mixed-W4/W8 QAT versus matched float-PTQ at both the 10% and 30% fit-subset scales, plus full-cohort uniform-W4 and mixed-W4/W8 pairs. The [baseline receipt](receipts/QAT_NATIVE_Q8_BASELINE_VALIDATION_R1.json) and [trained-pair receipts](receipts/README.md) record the actual artifact/source hashes; [the grader](../../../scripts/grade_qat_component_bootstrap.py) can consume additional matched endpoint directories.

## Estimand and resampling unit

This consumer compares fixed, complete deployment predictions on the same exposed F0 evaluation panel. The paired estimand is candidate minus reference for `t0`, `t1`, and their official equal-weight mean (`wp`), using the frozen deployment evaluator's per-component six-sum arrays. The bootstrap resamples the 141 source-qualified connected components with replacement, drawing 141 component IDs per replicate. Each replicate uses the same sampled component multiplicities for reference and candidate, pools each endpoint's six raw sums, scores both targets with the pinned `wnn.gmetric.wp_from_sums`, and records the paired difference. It uses NumPy `default_rng(20261013)`, 10,000 replicates, batches of 128, and NumPy's linear quantile interpolation for the 2.5th, 50th, and 97.5th percentiles alongside the full-fold point difference.

## Identity and mask checks

The grader requires the admitted R2 anchor manifest SHA-256 `8894af708dcba1500fc8c1b431d30468a29a9999531dd7133ec982c3b1d19556`, the exact ordered 468-group F0 membership, and the expected mapping to 141 components. It verifies all source-file hashes named by the anchor and each endpoint's own manifest. The endpoint manifests may have different hashes because they bind different parent checkpoints and exact export bytes; each must independently describe the same ordered evaluation groups and component mapping, and its source-file hashes must verify. The grader also requires `profile_only=false`, exact full-sequence replay, the official score metric, component-sum shape, and—against the native-Q8 anchor—the same selected-row total and exact per-component target-only sums (`W`, `WY`, `WYY`) for both targets. These statistics strongly check label/mask identity but are not a row-level mask digest; different masks with colliding per-component sufficient statistics are not ruled out mathematically.

Before grading a pair, the grader pools the anchor's component sums and checks `wp_from_sums` against the result receipt's stored `t0`, `t1`, and `wp` with absolute tolerance `1e-10`. This independently reproduced the native-Q8 point scores to within `1e-15`.

The first baseline-only pair uses recurrent W4 as reference and native Q8 as candidate. The full-fold WP difference is `+0.0008468`; its 10,000-replicate raw-sum bootstrap 95% interval is `[-0.0002493, +0.0022007]`. Target deltas are `+0.0013595` for t0 (interval `[-0.0001689, +0.0034892]`) and `+0.0003340` for t1 (interval `[-0.0006149, +0.0012818]`). These unchanged-parent quantization baselines are not trained QAT and the intervals are descriptive, not an automatic kill rule.

The trained uniform-W4 pair has candidate-minus-reference WP `-0.0001149` (95% interval `[-0.0009160, +0.0006608]`); the mixed-W4/W8 pair has `+0.0001390` (`[-0.0004814, +0.0007805]`). Both compare exact native QAT deployment endpoints against same-seed floating-parent PTQ endpoints on the same full F0 panel. Both intervals cross zero and remain descriptive component-resampling summaries, not stop rules or cross-seed evidence. The indexed [uniform review](QAT_10PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) and [mixed review](QAT_10PCT_MIXED_W4W8_MATCHED_GRADE_R1.md) audit their recipe and checkpoint ancestry separately.

At 30% subset coverage, uniform-W4 QAT minus matched float-PTQ is `+0.0002399` WP (95% interval `[-0.0003046, +0.0008922]`), and mixed-W4/W8 QAT minus matched float-PTQ is `+0.0002711` (`[-0.0002897, +0.0008545]`). These two within-scale pairs are matched at 4,380 updates and one seed; neither interval supports a precise sign. The 30% mixed receipt also records floating-parent mixed PTQ versus the frozen R2 native-Q8 endpoint, a contextual but unmatched diagnostic (`-0.0005725`, interval `[-0.0019464, +0.0007973]`). Comparisons between 10% and 30% absolute scores jointly change subset coverage and optimizer update count, so they do not isolate scale. See the indexed [30% uniform review](QAT_30PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) and [30% mixed review](QAT_30PCT_MIXED_W4W8_MATCHED_GRADE_R1.md).

At the 100% cohort, uniform-W4 QAT minus matched float-PTQ is `+0.0006216` WP (95% interval `[-0.0004012, +0.0016864]`), at 14,920 updates and one seed. The interval crosses zero. The [full-cohort grade](QAT_100PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) verifies the same training membership, sample order and normalization across arms, then checks exact QAT code/scale payloads against the native archive. The result is an exposed-F0, one-seed fixed-prediction comparison; it does not establish a precise effect or public transfer. Absolute scores across 10%, 30% and 100% fits jointly change cohort scale and update budget, so they do not isolate a scale effect. The full-cohort mixed-W4/W8 comparison is not covered by this grade.

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
