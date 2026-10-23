---
kind: review
status: complete
created_utc: 2026-10-22
updated_utc: 2026-10-22
hypothesis_ids: [H-PLATEAU-I1-HISTORY-20261020-R1]
experiment_ids: [E-PLATEAU-I1-20261021-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json
  - research/campaign_2026-10-21/next_cycle_r1/i1/runtime/panel_r2.log
  - artifacts/runs/i1_f2_s1002_b_r1/score.json
  - research/unified/reviews/H512_ENDPOINT_INTERPRETATION_REVIEW_20261021_R1.md
  - specs/plans/2026-10-21-next-local-kaggle-cycle.md
---

# Interpreting the i1 history standalone and fixed-mixture pattern

## Question and scope

Does the first F0 history contrast point to a useful representation contribution,
or could the fixed-mixture result come from prediction-distribution and
optimization effects? This is a read-only review of score and run metadata. I
did not open prediction or target arrays, fit tensors, or run code. F0 is
previously exposed development data; its component intervals condition on the
three fitted models and do not measure training, fold, or public-transfer
uncertainty.

## What the first cell shows

In the matched F0/seed1001 cell, B is the standard-input model without history;
C adds the frozen history channels; A masks both the i1 inputs and history.
C scores 0.655537 versus B at 0.659074, with C−B at −0.003537
(95% paired-component interval [−0.007283, −0.000066]). Yet the predeclared
`0.9*B + 0.1*C` endpoint is +0.000360 over B, with interval
[−0.0000119, +0.0007089]. Its point deltas are positive for both targets and
all four fixed position bins. The matching `0.9*B + 0.1*A` result is nearly
flat at −0.000023, with an interval spanning zero. The bin values are point
estimates only; no bin intervals were reported.

This pattern is compatible with a weak complementary correction: C can be a
worse full replacement while a small fixed contribution helps B. The contrast
with A makes the C mixture more interesting than a candidate-agnostic averaging
story. It still does not establish history complementarity. C−B is the clean
history treatment, but a better fixed mixture can also arise because C changes
prediction dispersion, clipping, or the optimizer's trajectory. F0 C's high
head-gradient share and clipping rate are descriptions of that run, not
evidence for a causal bottleneck.

There is a useful constraint on the shrinkage explanation. The official score
is a weighted Pearson correlation after clipping at ±2. A positive scalar
rescaling of an unclipped prediction alone cannot change Pearson correlation.
So a distribution-based explanation would need to change relative prediction
structure or interact with clipping; it cannot be only “C shrinks B.” The F0
summary does not report the clipped prediction fractions or covariance and
variance terms needed to tell those cases apart.

## What F2 can and cannot resolve

The available F2/seed1002 B run scores 0.684108 (t0 0.683801, t1 0.684415)
after 4,830 updates. That is a standalone fold-2 score, not a paired contrast;
its absolute value should not be compared with F0 B. At review time, F2 A and C
and the frozen mixture grade were still pending in the current run log. Until
they land, this cell says nothing about whether the F0 pattern recurs.

The frozen F2 A/B/C grade will provide the cheapest already-authorized
cross-cell check: inspect C−B and the fixed `0.9*B + 0.1*C`−B deltas for both
targets and the four fixed bins, alongside the A control. A recurring positive
fixed-mixture delta while C remains weaker alone would strengthen the case for
a useful small contribution. A null or negative mixture would leave the F0
pattern unconfirmed, not falsify every history representation. F0/1001 and
F2/1002 change both fold and seed, and both folds have prior exposure, so this
two-cell screen cannot separate fold from seed variation or provide fresh
confirmation.

Scores and bins alone still cannot distinguish complementary information from
distribution effects. If that distinction changes the next allocation, one
no-fit calculation on the already-saved prediction caches can decompose the
fixed mixture after the official clipping and label weighting. For each target
and fixed bin, let `B_c = clip(B, ±2)`, `M_c = clip(0.9*B + 0.1*C, ±2)`, and
`D = M_c - B_c`. Report the weighted covariance of `D` with the target and
with the target residual after its best weighted affine projection on `B_c`,
alongside the weighted variance terms and clipped fractions for B and M. A
positive residual covariance in both cells would support a distinct corrective
direction. A mixture change with little residual covariance but altered clipped
fractions or variance geometry would favor a distribution/calibration account.
This is diagnostic only: it uses exposed labels, adds no fit, and cannot prove
that the representation rather than its treatment-specific optimization path
caused the effect. No coefficient search is needed or proposed.

## Limits and records

The fixed C mixture's overall interval just crosses zero; the current F0 result
is therefore suggestive, not an established blend gain. The component
bootstrap addresses evaluation-unit variation conditional on fixed predictions,
not fitted-model variation. The F2 B score has no matched A/C result yet. No
public forecast or mechanism promotion follows.

Sources: [F0 A/B/C interim grade](../../campaign_2026-10-21/next_cycle_r1/i1/interim_f0_abc/RESULT_R1.json),
[current F2 run log](../../campaign_2026-10-21/next_cycle_r1/i1/runtime/panel_r2.log),
[F2 B score record](../../../artifacts/runs/i1_f2_s1002_b_r1/score.json),
[prior endpoint interpretation](H512_ENDPOINT_INTERPRETATION_REVIEW_20261021_R1.md),
and the [accepted cycle plan](../../../specs/plans/2026-10-21-next-local-kaggle-cycle.md).

## Source-based follow-up interpretation

After the fixed geometry diagnostic, root proposed and dynamics_builder independently checked a narrower optimization explanation using source and metadata only. Both cold B and C have identical stored initial parameters and normalization. B masks the 156 appended inputs after normalization, while C activates their existing random, nonzero input-projection weights. Their effective initial functions therefore differ. The cold treatment changes history exposure and its resulting optimization path together; it does not change parameter count or establish a measured conditioning/variance effect.

The [measured F0 geometry](../../campaign_2026-10-21/next_cycle_r1/i1/fixed_mix_geometry_r1/RESULT_R1.json) excludes clipping-only and pure scalar shrinkage accounts for its small positive point gain, while remaining conditional on the same exposed fits. It cannot determine whether the corrective direction arose from useful history information or the changed training trajectory. No added confidence interval or independent replication follows from this decomposition.

The already conditional strong-parent comparison offers a useful falsifier. Extend both control and treatment to the same 367-slot input, initialize appended projection weights to zero, preserve the original 211 input statistics, use one explicit shared normalization choice for new slots, and verify identical parent predictions before fitting. Then only the treatment receives unmasked history inputs and their gradients. Failure to improve the paired strong controls or the fixed endpoint contribution would weaken the explanation that a cold initialization/short schedule obscured useful history. This is an interpretation of the existing four-fit follow-up, not another admitted panel or a changed cold recipe.

The [warm-parent catalog](STRONG_PARENT_F0_F1_WARM_INIT_CATALOG_20261022_R1.md) records two necessary implementation gates: `none` to `i1_history` is currently rejected, and generic inheritance assigns new slots mean0/std1 while cold C fits their statistics. The separate 64-group i1 prior normalizes only the energy feature construction; it is not a replacement for those input statistics. Historical parent FP16/AMP and distinct initializer/cache ancestry remain explicit. Root still waits for the existing F2 cell before selecting a mechanism; no warm code or fits were added by this review.
