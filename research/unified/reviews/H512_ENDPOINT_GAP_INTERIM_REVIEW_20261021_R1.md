---
kind: review
status: running
created_utc: 2026-10-21
updated_utc: 2026-10-21
hypothesis_ids: [H-H512-ENDPOINT-GAP-20261021-R1]
experiment_ids: [E-H512-ENDPOINT-DIAGNOSIS-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json
  - research/campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json
---

# Reconsidering the H512 replacement result

## Question or claim

Why does H512 reproduce its earlier standalone score yet lose when replacing
TL3's student at the same fixed coefficient? The completed native endpoint
comparison is real, but it combines differences in fitting exposure, learning
objective, output scale, and serving precision. It does not isolate capacity.

## Mechanism and method

Root requested two independent source-only interpretations from `recovery_audit`
and `dynamics_builder` (native Luna/max). Neither opened prediction arrays or
ran models. They checked the exact archives, wrapper, coefficients, replay
coverage, source-qualified membership and existing recipe audits. The reviews
are independent inspections, not independent statistical replications.

The new [diagnostic contract](../../campaign_2026-10-21/next_cycle_r1/ENDPOINT_DIAGNOSIS_PLAN_R1.json)
fixes covariance, clipping, prediction-scale, differential-parity and sequence-position
checks on saved predictions. A separate eight-component native precision probe
will compare the existing H512 checkpoint in FP32 and dynamic int8. Neither
analysis fits blend coefficients or admits a new training run.

## Evidence and interpretation

The [complete native grade](../../campaign_2026-10-21/next_cycle_r1/NATIVE_GRADE_R3.json)
puts original TL3 at 0.70181812, the CP5 replacement at 0.69445462, and the SWA345
replacement at 0.69444908. The replacement losses are approximately 0.00736;
SWA minus CP5 is only -0.00000553. Both targets decline. The 468 sequences and
141 evaluation components are exposed, and the bootstrap conditions on these
fixed predictions.

The source review found the same native v3 binary and backend for original and
replacement replays, matching row/mask coverage, and unchanged wrapper, retained
members and coefficients. R3 explicitly binds the repaired NPY-header guard;
R2's earlier failure remains historical. These checks weaken a simple package,
ordering or evaluator explanation. They do not measure H512-specific int8 error.

A more consequential comparison difference is now traceable. The original
`weights_s5.npz` hash exactly matches the older `w18final_distilled_s5@ep1`
production payload. Its preserved final-training script explicitly uses supplied
train and validation data, includes held-out-valid rows, and labels its final-fit
metrics in-sample. H512's recorded fit excludes the 468 F0 evaluation groups.
The [identity and source receipt](../../campaign_2026-10-21/next_cycle_r1/TL3_STUDENT_EXPOSURE_RECOVERY_R1.json)
binds the payload and script hashes. Complete row-group execution receipts and
recursive teacher/initializer identities are still missing.

The old student also had a different task: it continued an existing parent with
70% teacher supervision, heterogeneous family targets, propensity weighting and
auxiliary objectives. The replacement was a cold WMSE model. Reproducing the
latter's score says little about whether it can replace the former's role in an
ensemble. Prediction scale matters here: positive rescaling can preserve a
standalone Pearson score while changing a member's effective contribution to a
fixed weighted sum. Its size and direction must be measured in these caches.

## Measured saved-prediction checks

The [streamed R3 diagnosis](../../campaign_2026-10-21/next_cycle_r1/endpoint_diagnosis/RESULT_R3.json)
reproduces all three native endpoint scores within1e-15. Standalone SWA improves
H512 by0.00022538, yet its native endpoint changes by−0.00000553. The replacement
raises target-weighted clipped-prediction standard deviation by1.07%/1.48%, while
label covariance changes−0.185%/+0.613%. This algebra explains the correlation
loss as a variance cost exceeding covariance benefit; it does not establish a
calibration treatment or an optimal blend weight. H512's weighted partial target
correlations conditional on TL3 are negative(−0.1615/−0.1249), descriptive on the
same exposed predictions.

Every fixed position bin loses. The earliest CP5 delta is−0.016978 with conditional
component95%interval[−0.027560,−0.007647]; the latest is−0.005931,
[−0.008317,−0.004246]. Only17,198of808,863scored rows lie in the earliest bin,
whereas412,135lie in the latest. Pooled correlation is not an additive average
of bin correlations, but these results weaken a pure early-state explanation.
The bootstrap captures evaluation-component variation conditional on fixed fits,
not training variation, a fresh holdout, or public transfer. Its resampling
population is all141fixed evaluation components, including zero contributors in
a given bin; it does not condition on that bin having observations in a component.

The [feature-only precision probe](../../campaign_2026-10-21/next_cycle_r1/NATIVE_PRECISION_EXECUTION_R1.json)
finds native FP32 H512 within1.10e-6RMS of Torch, versus0.002655RMS for current
int8 serving. Its CP5 endpoint reconstruction passes, but original TL3 reconstruction
fails the unchanged1e-5gate(max0.002090), so it stops before label access. No
precision score or conclusion about quality is available. The immutable outputs
and source-only diagnosis preserve this unresolved adapter issue.

## Limits and validity

Final production fitting on supplied validation data is permitted; the exposure
difference changes the meaning of this local comparison. The negative
user-reported public result remains separate evidence against this particular
replacement. Exposure asymmetry cannot explain it away. The two retained tied
parents still lack exact fitting receipts. Small native probes diagnose numerical
behavior; their selected-panel scores cannot establish a new submission gain.

The [older TL3/R3 diagnostic](../../campaign_2026-10-18/online_features_r1/tl3_diagnostics_r1/README.md)
already showed why prediction disagreement alone is insufficient: its residual
errors correlated about 0.984 and its fixed mixtures lost. The present analysis
must measure conditional target information rather than repeat a diversity claim.

## Decision and next step

Keep the frozen i1 A/B/C screen running. Judge it by matched cells, not by its
absolute score against TL3. Diagnose H512 native precision and ensemble scale
before interpreting the fixed replacement loss as an architectural result.
Keep the planned strong-parent continuation comparison as the production gate;
a mechanism that helps a small cold model must also help the stronger clean
control. No width grid, blend search or new production fit is admitted here.

## Source records

- [Exact original archive entries](../../campaign_2026-10-20/submission_delivery_r1/tl3_member_adaptation/SOURCE_ARCHIVE_CONTENTS_R1.json): immutable member hashes.
- [Older candidate manifest](../../campaign_2026-10-06/bandit_continuation/reviews/maxim_replication_20261007/candidate-manifest.json): matching seed-5 payload and final-training description.
- [H230 recipe audit](../../campaign_2026-10-06/bandit_continuation/wide_baseline_heterogeneous_H230/H230_MAXIM_RECIPE_AUDIT_V1_20261008.md): exact source command, target construction and unresolved teacher ancestry.
- [Current frontier](../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md): owner, resource allocation and execution status.
