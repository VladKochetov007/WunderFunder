---
kind: result
status: complete
created_utc: 2026-10-24T13:42:00Z
updated_utc: 2026-10-24T13:42:00Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1, H-ASTRA24-FUND-01]
experiment_ids: [E-20261024-fixed-direction-probe-r2]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_RESULT_R2.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/direction_probe_r1/EXECUTION_R2.json
---

# What explains the common online-correction direction?

## Question or claim

The F0 improvement from a fixed, small correction is not explained by ordinary affine output calibration. The richer B/C corrections mostly share the direction already present in mean-only A. Their small remaining score derivatives are uncertain. F1's A correction has a much weaker local slope and still gives essentially zero measured gain at the frozen coefficient.

This is a decomposition of existing predictions, not a causal attribution to input features or training choices. In particular, A contains history, ongoing state updates and trained head capacity; the missing equal-capacity zero-state Z control remains necessary to distinguish those explanations.

## Mechanism and method

The [frozen R2 recipe](DIRECTION_PROBE_RECIPE_R2.json) uses the existing parent `p` and each saved correction `q−p`. It calculates the right derivative of the official clipped, label-magnitude-weighted correlation along `p + alpha*(q−p)` at alpha zero and 0.1. It projects the alpha-zero clipped tangent onto four signed calibration tangents, and for B/C also onto the common A tangent. Target-specific weighted calculations are averaged only at the end.

F0 has five unique endpoints, 468 source-qualified groups, 141 dependency components and 808,863 scored rows. A lag-16 is an exact duplicate of A lag-1. F1 has A only, 457 groups, 149 components and 812,167 scored rows. Fixed alpha-0.1 endpoints retain the previously frozen float32 arithmetic. All parent, raw and shrunk official score fields reproduce the earlier grades exactly. No coefficient was searched or exported; no inference or fitting ran.

## Evidence and interpretation

Derivatives below are WP change per unit alpha, **not measured endpoint gains**. Intervals are conditional 95% component-bootstrap intervals from 2,000 draws. All endpoint point estimates are shown, without selecting a new winner.

| Fold / endpoint | Derivative at alpha 0 | Derivative at alpha 0.1 | Remaining derivative after calibration + A | Conditional interval for remaining derivative |
|---|---:|---:|---:|---|
| F0 A | +0.00157701 | +0.00105675 | Not applicable | Not applicable |
| F0 B lag 1 | +0.00157766 | +0.00104698 | −0.000002714 | [−0.00009650, +0.00008704] |
| F0 B lag 16 | +0.00157463 | +0.00103792 | −0.000014617 | [−0.00013037, +0.00008166] |
| F0 C lag 1 | +0.00163075 | +0.00110014 | +0.000051219 | [−0.00003448, +0.00014486] |
| F0 C lag 16 | +0.00155948 | +0.00102702 | −0.000010559 | [−0.00015365, +0.00013144] |
| F1 A | +0.00015110 | −0.00024195 | Not applicable | Not applicable |

The calibration span accounts for 67.0–70.7% of F0's centered weighted correction variance, but its projected score derivative is only about 0.00000127, less than 0.1% of the total F0 derivative. Weighted correlation is largely insensitive to affine shift/scale away from clipping. A large variance share therefore does not identify the source of score improvement. This narrows the simple-calibration explanation while leaving generic nonlinear head capacity and adaptive information unresolved.

Adding A to that span accounts for 98.79–99.58% of B/C's weighted tangent variance, across their target-specific calculations. The remaining derivatives are small and mixed; every displayed interval crosses zero. This does not establish equivalence or make relationships useless. It shows why the present endpoints cannot support a strong claim that the extra relationship terms are responsible for the shared improvement.

For A, F0's alpha-zero derivative is +0.00157701 with interval [+0.00004143, +0.00333659]. F1's is +0.00015110 with interval [−0.00093402, +0.00137125]; at alpha 0.1 it is −0.00024195 with interval [−0.00131307, +0.00095809]. These are conditional local slopes, not an instruction to tune alpha. The actual fixed-alpha differences remain **+0.000131368 in F0 and −0.000004696 in F1**.

Clipping is sparse at the parent: F0 has 1/7 saturated scored predictions for targets 0/1, and F1 has 41/34. No evaluated row is exactly on a clipping boundary. Full-sample calibration ranks are two, increasing to three with A. Some component resamples omit the rare saturated rows and reduce rank; rank histograms and every interval's valid-draw count are retained in the result. These changes are reported, not silently regularized away.

## Limits and validity

Both folds are exposed development evidence already used for endpoint grading. Fold and fitting seed change together. Intervals condition on the fitted endpoints, evaluated components and prior selection; they omit training, selection, fold and public-transfer uncertainty. Projections use evaluation-label weights and describe a tangent span, not a deployable calibrator. General clipped tangent combinations need not be realizable by the same linear combination before clipping. Derivatives use real arithmetic; fixed endpoints use the frozen float32 recipe.

The incomplete historical twelve-fit panel remains incomplete. This run does not create clean cross-validation, establish a new best, or forecast a public score. The [independent result review](RESULT_REVIEW_R2.md) reproduces pooled moments, derivatives and projection ranks. Its independent bootstrap interval reconstruction is incomplete after reviewer-script errors; the intervals above remain actor-produced, conditional diagnostics.

## Decision and next step

Prioritize the prospective matched Z-versus-A screen before expanding relationship complexity: equal 419-wide head capacity, zeroed extra state versus actual A state, two source-qualified fold/seed cells, four total small-head fits, reused exact parents and no new teacher/student fits. That screen remains unadmitted. Preserve C lag-1's uncertain residual signal as a follow-up possibility, not a selected serving rule. Avoid treating common shrink gains or high calibration variance shares as proof of a particular mechanism.

## Source records

- [Result JSON](DIRECTION_PROBE_RESULT_R2.json), SHA-256 `cf7dbcc3da251fd77fff4a498115be9d12709fd293667c12c98ca507bef52413`, retains all target-specific slopes, intervals, projection diagnostics, clipping transitions and exact-score checks.
- [Component moments](DIRECTION_PROBE_COMPONENT_MOMENTS_R2.npz), SHA-256 `84c01f89f4c1426aebe64c375294cdf4475416424e95db4a13344b3b183539a9`, 2,359,790 bytes, contains full additive and stable centered tables for independent reconstruction; it is derived exposed-label evidence.
- [Root admission](DIRECTION_PROBE_ADMISSION_R2.json), [source manifest](DIRECTION_PROBE_SOURCE_MANIFEST_R2.json), [actual producer preflight](DIRECTION_PROBE_PREFLIGHT_R2.json), and [independent contract review](CODE_REVIEW_R2.md) bind the exact inputs and consumer behavior.
- [Execution receipt](EXECUTION_R2.json): 90.434 seconds, one CPU core, 399,998,976-byte peak, zero OOM events and zero swap; 12,028 memory-max reclaim events show that the memory cap was active. The [execution log](EXECUTION_R2.log) is empty on success. Pre/post target and all eight prediction hashes agree.
- [Astra fundamentals report](../../astra_fundamentals_r1/FUNDAMENTALS_AND_MISSING_TESTS_R1.md) supplied the question. [R1 failure](EXECUTION_R1.json) and [its exact snapshot](r1_snapshot/README.md) preserve the recipe-key failure before any data values were read; it was execution evidence, not a negative scientific result.
