---
kind: review
status: complete
created_utc: 2026-10-23T17:17:50Z
updated_utc: 2026-10-23T17:28:29Z
hypothesis_ids: [H-ASTRA23-09-D, H-ASTRA23-08-D]
experiment_ids: [E-ASTRA-ULTRA-AUDIT-20261023-R1]
evidence_paths:
  - research/campaign_2026-10-21/next_cycle_r1/WEIGHT_MEAN_PROJECTED_GRADE_R1.json
  - research/unified/reviews/astra_ultra_audit_20261023_r1/09_research_value_and_delivery.md
  - research/unified/reviews/astra_ultra_audit_20261023_r1/08_transfer_and_information_limits.md
---

# Corrections and interpretation boundaries for the independent reports

The original reports are preserved as each reviewer submitted them. Root's
source checks and design criticism below control the synthesis and allocation
recommendations. They are documentary/algebraic checks, not newly executed
scientific experiments. Independent model reviews can repeat stale summaries or
propose insufficient controls; agreement is not a substitute for these checks.

## R01 — The projected checkpoint mean was already graded

[Report 09](09_research_value_and_delivery.md), including card H-ASTRA23-09-D,
says the projected mean still needs export and grading. This follows the older
[packet README](../../../campaign_2026-10-21/next_cycle_r1/weight_mean_projected_r1/README.md).
The newer [campaign frontier](../../../campaign_2026-10-21/next_cycle_r1/FRONTIER.md)
and [completed grade](../../../campaign_2026-10-21/next_cycle_r1/WEIGHT_MEAN_PROJECTED_GRADE_R1.json)
show that this work is done:

| Frozen comparison | Observed difference | Conditional component-bootstrap 95% interval |
|---|---:|---:|
| Projected mean minus WMSE CP1 | +0.0000415565 | [−0.0003412332, +0.0004597788] |
| 75% oldZ4 / 25% projected mean minus 75% oldZ4 / 25% WMSE | +0.0000109615 | [−0.0000830101, +0.0001132356] |

This is one exposed F0 comparison, 468 groups / 141 components, with the same
ancestry limitations as the original grade. The frontier records 157.42 seconds
for export; the grade records 15.0999 seconds. The review's proposed future
1,000–3,000-second estimate must not replace those historical measurements.

**Disposition:** do not repeat the export/grade merely to complete this card.
Register it as a duplicate already evaluated comparison with a tiny, uncertain
increment. The averaging family remains open to materially different, explicitly
justified recipes. This correction does not overturn report 09's main proposal
to recover an exploratory contrast from completed online heads.

## R02 — A fixed derangement is not an information-destroying flag control

[Report 08](08_transfer_and_information_limits.md), card H-ASTRA23-08-D,
proposes correct per-field censoring flags versus a fixed deranged mapping into
an otherwise unrestricted dense residual head. The proposed censoring distinction
is plausible, especially after age saturation. The proposed control does not
identify its interpretation: for permutation matrix P and freely learned first
layer W, W(Pz) can express the same functions as an unconstrained first layer
on z by reindexing its columns. The flags and their identities remain recoverable.

**Disposition:** retain the censoring question, redesign the correspondence
test before admission. A comparison against no flags can test incremental value.
A restricted shared field-pair encoder, with an explicit raw-input bypass, can
test a particular correspondence prior. A sorted/pooled flag summary can remove
identity while retaining occupancy information, but changes the representation
and its learning geometry. None is automatically a perfect causal control.
Report 02's restricted paired encoder and report 07's sorted-age comparison
explain this distinction. No benefit or failure has been measured for this card.

## R03 — Identical themes do not imply identical experiments

History-versus-second-seed cards 01-A, 06-A and 09-B ask a similar allocation
question. Card 01-A additionally repairs the initial-function confound with a
zero-extended warm starting function; cards 06-A/09-B principally describe the
existing cold recipe and its ordinary-seed comparator. Their old anchors cannot
silently become matched controls for a changed initializer or update schedule.
Freeze one version and name the reused artifacts before counting new fits.

Cards 03-D and 05-B revisit the already documented S14 credit-horizon direction.
Its historical FP16 engineering profiles completed; its scientific comparison
did not. A strict-FP32 future recipe is an explicit descendant, not a faithful
rerun of an already positive quality experiment.

## R04 — Several tempting explanations are already narrowed by the record

The [evidence map](EVIDENCE_MAP.md) provides the primary links and exact limits:

- EV17: warm zero-extension initial parity is complete on the selected sequences;
  paired warm quality is still missing. Do not repeat a stale pending-parity claim.
- EV22: the exact H512 F0/seed1003/CP5 score was reproduced to about 1.3e−9.
  A single-cell-versus-four-fold-mean comparison had produced the earlier false
  non-replication story. This does not make H512 the stronger submitted endpoint.
- EV05/EV15/EV16: runtime and sampled precision discrepancies are much smaller
  than the relevant endpoint losses; exposure and endpoint composition still
  prevent a clean architecture conclusion. Negative public observations remain.
- EV21: matched warm KD protects much more performance than its small increment
  over an already-distilled initializer suggests. This is one historical warm
  FP16 recipe, not a fresh FP32 or cold-start mechanism demonstration.
- EV20: the current teacher-cache branch bypasses generic loss dispatch. The
  historical NLL producer did implement NLL; its distinct source must be retained.
  Current flag semantics cannot retrospectively invalidate that experiment.

## R05 — Costs, partial panels and mathematical coefficients

Seven saved online heads do not complete the twelve-head frozen panel. A future
grade of the complete F0 subpanel would be a separately declared exploratory
analysis. It must preserve both lag contrasts and the original incomplete status.
The repeated lag-independent controls suggest possible exact reuse; equal
prediction hashes alone do not establish all source/membership requirements.

The online variance warmup observation λ^83 ≈ 0.894, with λ = 2^(−1/512), is the
remaining coefficient on its initialized variance term. It is not automatically
89.4% of the final variance or an effective sample-size calculation. The broader
point is that a short warmup and a continuing estimator can differ in estimator
maturity even under stationary relationships.

Zero output initialization preserves the starting prediction of a residual head.
It does not equalize its parameter derivatives, input rank, regularization or
optimization trajectory. Likewise, zero new fits does not mean zero inference,
export, I/O, grading, or engineering cost. All proposals remain unexecuted here.

## Source records

- [Independent reports and contract](README.md)
- [Verified historical evidence and limitations](EVIDENCE_MAP.md)
- [Primary literature register](../../sources/ASTRA_ULTRA_REVIEW_SOURCES_20261023_R1.md)
