# TL3_DIAGNOSTICS_R1 — adversarial skeptic review

**Verdict: ITERATE for any quality or transfer claim.** Retain the result as a
valid descriptive score of fixed caches on exposed F0. No public-gain or
clean-CV claim follows. This review author also wrote and ran the grader, so
this is an adversarial interpretation pass, not an independent implementation,
regrade, or approval of the grader.

## Findings

### 1. Score and cache identity: no concrete mismatch found

The recipe pins the unchanged `global_wp_clip2_v1` metric, official cached mask,
warmup cutoff 99, source-qualified F0 group order, and two fixed mixtures
before scoring. The preflight binds 468 groups, 141 components, 808,863 selected
rows, the target-cache SHA, prediction/sidecar hashes, and exact TL3 and R3 ZIP
hashes; it confirms finite F0 predictions and sequence-ID alignment for TL3
and R3. The full TL3 replay receipt says the unchanged three-member ZIP was
run through all 468 groups without reading labels or the scoring mask. The
scored R3 q16x value, `0.681423403765`, matches its separate full-F0 serving
receipt to below `1e-15`. These checks support arithmetic and identity for
this diagnostic panel, not model ancestry or predictive validity.

The exact pins are TL3 ZIP `d80e69d3cb52d8ee45158fc8d0f3a8af232b029b07ca5b313c61cffef029e569`
and prediction `3a7456f0a2e294a852618535b9e3f2551fc15c5b70fd033824bddb10be83bbe3`;
R3 ZIP `27f01f53ba32f7256ce1663e8b163b382e673fc96e7870fdebdf310d4886f98e`
and prediction `e82e1a9844b323b9886927adab4d3d579a6f278654eef322488de7784d4823e6`;
parent checkpoint `78357eca2aec188e227544d46887935265720c35e99a1437269b952d01eb88c2`.
The fold0 split is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`.
The frozen grader source SHA is `d7b39aec6166a2496272beeca5a16497f9d7abcaabfe1b3bf4e7b415e4b567dd`;
it hashes the official metric, alignment and bootstrap helpers in preflight.

Evidence pins: [preflight](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/preflight.json),
SHA-256 `6991d856b6528af95d75f13904cdd1b13fcbe5e15a4a8d10ef0b031439289722`;
[recipe](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/recipe.json),
`30296eb5f7e7a8e608d3e181c1d6bfd43074b8dae8ff056f6967387b695d7f76`;
[result](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/result.json),
`10a42eefcda3d6703ddee2006ebfc7bd69915830d34b9cfcf0a160dad8ae9451`;
[TL3 full-F0 replay](../../campaign_2026-10-18/online_features_r1/tl3_replay/full_f0_serving.json),
`1f7b1251fd10c9c5c24e576f022238c32b6948d5ce9d8fc0e4c066eac3f7af25`;
[R3 full-F0 serving receipt](../../campaign_2026-10-18/geometry_variations_r3/geometry_r3_reverse_half_q16x_20261018_v2_f0_serving.json),
`b28c57cef8a316a21b5db919924e4f165bb56dc19c0215c282d8d142b333c7c6`.

### 2. TL3 ancestry and F0 exposure block transfer interpretation

TL3 scores `0.701818121293` against parent `0.681226513222`; the conditional
component interval for their difference is `[+0.01525849,+0.02705575]`. But
the [data-sentinel audit](../../campaign_2026-10-18/online_features_r1/DATA_AUDIT.md),
SHA-256 `58120576f89cc3e737493ebcf543758b534951823b7bfc0678b7409a383f9469`,
and [TL3 import record](../../campaign_2026-10-18/geometry_variations_r3/TL3_IMPORT.json),
SHA-256 `95804c5be4495d8c9a2b2b622f85a63a387ae5a82e6db8b22e00585a3baee020`,
say the three exact members are `distilled_s5@ep1` (0.34),
`fttall_tlead_s5@ep3` (0.33), and `fttall_tlead_s6@ep3` (0.33); the embedded
description calls the parent teachers all-data, while actual student,
teacher, initializer, and sequence membership remain unknown. This leaves
possible F0 fitting overlap unresolved. F0 is also repeatedly exposed and used
for model/alpha/geometry choices. The large positive interval therefore
describes the observed caches only; it cannot establish a generalization gain.

The warm parent is a component-clean F0 reference with matching F0 teacher
membership according to the machine [data-audit receipt](../../campaign_2026-10-18/online_features_r1/data_audit_receipt.json),
SHA-256 `ebb60da769994cc830521e743d583effcaa7bcf5fd98e14424c541db583d4a52`;
its F0 evaluation is still exposed.
R3 is not a new fit: the exact q16x ZIP is an inference-only `alpha=-0.5`
derivative of R2, selected after inspecting F0. Thus both the overall panel
and the R3 endpoint carry selection exposure, although the two TL3 mixture
weights themselves were fixed before the new score was read.

### 3. Fixed blends are honestly fixed, but do not show a local advantage

The 75/25 and 50/50 TL3/R3 scores are `0.698202379728` and
`0.693739652639`, respectively. They are `−0.00361574` and `−0.00807847`
against TL3, with conditional component intervals `[-0.00489220,-0.00257992]`
and `[-0.01083477,-0.00583761]`. This supports the narrow statement that
neither mixture beats the TL3 cache on these F0 rows. It does **not** prove the
mixtures cannot transfer: an R3 component might regularize a TL3 endpoint that
is overfit to F0. There is presently no evidence that it does so, and these
scores do not support choosing either blend over TL3.

The component intervals resample 141 F0 components conditional on fixed
predictions. They do not include fitting-seed variation, TL3 ancestry
uncertainty, F0 selection, or public-transfer uncertainty; they are not
selection-adjusted comparisons across endpoints. R3-minus-parent is only
`+0.00019689`, CI `[-0.00002585,+0.00043871]`, a small exposed-panel difference
whose interval crosses zero, not evidence of equivalence or a transfer failure.

### 4. Correction geometry is not a causal mechanism

TL3 and R3 parent-relative corrections have label-magnitude-weighted
covariance/correlation `−0.002153/−0.6018` and `−0.002087/−0.6325` for targets
0/1. Their residual-error covariance/correlation is `+0.954686/+0.9836` and
`+0.942230/+0.9844`. Opposite correction directions therefore coexist with
strongly aligned errors on this panel; the blends' lower scores than TL3 agree
with the absence of an exposed-F0 blend win. Neither covariance nor correlation
establishes causal complementarity or predicts transfer. The preceding
[factor-geometry packet](../../campaign_2026-10-18/online_features_r1/prediction_geometry/README.md),
SHA-256 of its receipt `4978d0b88ea640ed3ba5bf40bbee0e79406b4b88e1d5e076d7e7c0db96067c0e`,
covered five adapter caches, not TL3, so it supplies no independent TL3
mechanism confirmation.

## One useful next test

Run one fixed-coefficient blend-versus-TL3 robustness grade on the canonical
F2 evaluation set (443 source-qualified sequences) in the [prepared F2/F3
packet](../../campaign_2026-10-06/bandit_continuation/ops_continuation_20261011/h382_local_clean_crossfold_f2_f3_r1/README.md),
SHA-256 `b6f3d512d1e00287278eec3da397d9481714da784d816fdcd30144139b44b751`:
replay the exact TL3 and R3 ZIPs and score the two unchanged mixtures against
the standalone TL3 endpoint with `global_wp_clip2_v1`. First bind the TL3
members' student/teacher/initializer sequence manifests to F2's split; if that
ancestry cannot be recovered or overlaps F2 evaluation components, mark
TL3-containing grades potentially fitted. F2 was inspected earlier, so even a
disjoint result is cross-fold robustness evidence, not a pristine holdout.
This tests whether the F0 blend penalty repeats on a different
source-qualified fold without tuning weights; it cannot by itself establish
public transfer.

## Scope and source record

Data-sentinel and factor-geometry prerequisites were completed before this
review. This pass read metadata, receipts, and source only; it did not read
target or prediction payloads, rerun scoring, fit, or modify the recipe or
grader. The evidence chain begins at [TL3_DIAGNOSTICS_R1](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/README.md).
