---
kind: idea
status: proposed
created_utc: 2026-10-15T16:29:33.670324+00:00
updated_utc: 2026-10-15T16:29:33.670324+00:00
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015]
experiment_ids: []
evidence_paths: [../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md]
---

# Is KD adding information or preserving the inherited function?

## Question or claim

The matched alpha0/.9 continuation removes both the external teacher signal and
a prediction constraint. If KD wins, it still does not establish that the new
teacher cohort teaches a better function. A fixed-parent prediction control can
separate useful new target information from preserving the starting solution.
This is a new prospective control; current CP5 grading has not been read here.

## Mechanism and method

Use the identical clean F0 warm parent and the newly fixed two-epoch/CP1,
seeded-order C control from the shared mechanism wave. Add S: alpha.9 teacher
mixture with the exact frozen parent's causal main predictions replacing only
the external T3 predictions. Keep fitting membership, weighting, auxiliary losses,
optimizer, horizon, normalization and FP32 grading fixed. This still has prior
teacher ancestry through the initializer; never call it teacher-free.

At initialization the unscaled parent-output matching term has zero gradient
(up to inference precision), so subsequent optimization is constrained around an
already useful function. This transfers a mechanism from
[initializer-preservation literature](../sources/INITIALIZER_PRESERVATION_SOURCES_20261015_R1.md),
not those papers' empirical claims. Compare S−C and both against their parent.
A useful S is eligible as a competition recipe even if the interpretation remains
uncertain. If S is materially worse than T3 under a fair longer comparison,
parent preservation alone does not explain the teacher's value in that recipe.
If S matches or improves T3, prioritize useful teacher residuals or stronger
initializers instead of simply adding homogeneous KD epochs.

## Evidence and interpretation

CP1 gives A0−A9 −.00394686 on exposed F0/seed3. It motivates this discriminator,
but final CP5 and seed variation remain necessary. The historical older parent
has a higher point grade and twice the optimizer attempts of the current parent;
[the audit](../../campaign_2026-10-15/student_quality_tradeoff_r1/PARENT_UPDATE_BUDGET_AUDIT_R1.json)
prevents attributing that difference to teachers alone. Existing QAT self-distill
results concern a different precision/serving treatment and do not answer this
float-training control.

## Limits and validity

Raw parent targets and the current0.8125-scaled T3 cache differ in calibration
as well as information. A positive recipe result is useful, but an informational
claim needs a separate fitting-only calibration control (freeze any per-channel
scale/offset before evaluation). Do not optimize calibration on exposed grades
and present the result as a prespecified causal effect. H-195's relational
regularizer is distinct from this pointwise parent-output reference.

A source trap: `FastEMA.update_parameters` copies the **post-update** learner on
its first call. Therefore `--ema-decay 1` does not exactly freeze the initializer;
it freezes after the first update and carries a mixed-weight hidden history on
that first sequence. Do not label this shortcut an exact parent-target control.
Reuse an actual fixed-initializer inference path with fold-appropriate validation,
or build a provenance-bound parent prediction cache. The existing final-adaptation
flag cannot simply be enabled for held-out recipe comparisons because its source
contract is intentionally all-data-only. Preserve ordinary EMA behavior.

## Decision and next step

No fit or new implementation is admitted here. After current grades and the
already-reviewed profiles, prepare one S arm beside the shared C control if the
expected information justifies it; don't silently change the C/G/F/W wave.
A two-epoch S fit has the current roughly37-minute H256×2 cost basis; profile any
new fixed-teacher forward path. A fitting-only FP32 cache needs about1.91GB plus
metadata; extrapolating the100-second/1,873-sequence validation export suggests
roughly10–15 GPU minutes, provisionally, with different train I/O. Stream to
physical disk and admit actual memory/disk usage first. Check complete-sequence
teacher chunk/reset parity and exact deployed target clipping/rounding, then
export all1,873 valid sequences and grade only the468 excluded groups. Root owns
allocation; source/recipe review and independent grading remain separate.

## Source records

[Current trajectory](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md),
[shared control wave](../../campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_WAVE_R3.md),
[source methods](../sources/INITIALIZER_PRESERVATION_SOURCES_20261015_R1.md),
[trainer](../../../scripts/train_rnn.py).
