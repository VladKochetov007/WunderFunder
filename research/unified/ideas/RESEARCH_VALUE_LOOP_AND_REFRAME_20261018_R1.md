# Research value loop and fresh reframe — 2026-10-18

## Decision question

The user clarified that a public submission is a confirmation of transfer, not
the research objective. The objective remains improving the expected score of
our strongest complete, deterministic endpoint under the unchanged
`global_wp_clip2_v1` metric. Local component and training-seed evidence drives
which model or mixture we prepare; a public score checks whether that direction
transfers.

The useful unit of research is a **decision changed by evidence per expensive
GPU-hour**. GPU utilization by itself is not progress. A fit earns its cost when
it answers a frozen comparison, preserves enough per-run evidence to diagnose
why it moved, and has a clear route into the final teacher, student, or served
mixture.

## What we need to earn a larger score

1. **A measurable source of edge.** Each experiment should isolate one mechanism
   with an exact control, causal inputs, source-qualified fitting membership,
   and the official scorer. This distinguishes useful signal from leakage,
   evaluation noise, and model capacity.
2. **Evidence that survives training and evaluation variation.** Keep the
   registered paired seed/fold comparisons, report fold-level and per-seed
   effects, and resample connected components separately. Existing folds are
   exposed development evidence; more seeds do not make them pristine.
3. **A complete-endpoint contribution.** A stronger teacher, a useful feature,
   or a better student is an intermediate. Measure the resulting deterministic
   student or fixed blend against its matched complete endpoint. Do not infer
   that effects add, or that a better research-only model is already a better
   submission.
4. **Serving evidence before release.** Check causal state reset, exact
   preprocessing parity, runtime, size and deterministic output under the
   competition callback. A research feature without a streaming implementation
   is evidence to invest in that path, not a package candidate.
5. **One transfer check at the right time.** After local evidence supports an
   endpoint, submit it to see whether the public slice agrees with the local
   ranking. A mismatch is a research question; it is not a reason to tune to
   leaderboard noise or to add a local delta to a public score.

Use CPU for bounded data audits, sequence summaries, source and recipe checks,
and score aggregation. Use GPU for paired learned comparisons whose outcome
could change architecture, feature or distillation choices. Prefer streaming
over staging every sequence into a multi-hundred-gigabyte feature array. Keep
failed fits, ambiguous comparisons and negative controls in the evidence trail;
they improve the next allocation even when they do not improve the score.

## Current high-information work

The current highest-value spend is the registered R4 comparison: 30 matched
H512×1 GRU pairs across four component folds, with the 60-channel
geometry/discrete suffix present versus zeroed. It directly asks whether the
features that showed conditional signal in tiny pilots remain learnable at a
much stronger from-scratch capacity. It does **not** test a teacher/student
interaction, serving parity, or improvement over the strongest complete local
endpoint.

Evidence motivating R4 is deliberately mixed. The R1 moment pilot's pooled
gain was opposed across exposed folds. The tiny R2 screen found heterogeneous
effects and harmful naive concatenation in some arms. R3's three-seed H128×2
screen put the all-feature arm below control, while a fixed half control /
dictionary mixture was higher on only 46 evaluation sequences. These results
justify the scale test; they do not establish a universal feature effect or
reject the family. R4 is fixed to run all 30 pairs even if early scores look
weak. Interpret the result at its actual scope:

- A positive R4 result supports a stronger follow-up that tests the features
  inside a useful teacher/student or complete-endpoint path; it is not by itself
  submission evidence.
- A mixed or imprecise result should guide a targeted representation,
  interaction, or replication question. An interval crossing zero alone does
  not close the direction.
- A consistently negative result can defer this exact standalone concatenated
  H512 recipe while preserving possible value in a teacher, student target,
  residual, gate, or fixed mixture.

All four folds are exposed or potentially exposed. The 30-pair result can
support a strong local development comparison, but not a pristine holdout or a
public score forecast. The running recipe and its admission status are tracked
in the [R4 validation note](../validation/GEOMETRY_FROM_SCRATCH_R4_20261017.md).

## Fresh proposals after repository overlap checks

The independent reframe proposed five literature-inspired directions. Its
priors were unknown and its first pass did not inspect this repository, so each
idea was checked against existing work before allocation.

| Proposal | Literature analogy | Existing overlap and next useful question |
|---|---|---|
| Lagged cross-instrument information (`i1` to `i0`) | Hasbrouck's trade/quote VAR and Cont et al.'s order-flow imbalance work ([Hasbrouck](https://doi.org/10.1111/j.1540-6261.1991.tb03749.x), [Cont et al.](https://arxiv.org/abs/1011.6402)) | Not a new family: E-334's DTW lead/lag ridge was about −0.00008; E-415 proxy and shifted controls did not improve the unchanged backbone; E435 already proposed an `i0`-lag / shifted-`i1` control. Reopen only with a materially different lag representation or conditioning mechanism and a new placebo. |
| Causal per-sequence normalization | RevIN studies instance normalization under time-series shift ([ICLR paper](https://openreview.net/pdf?id=cGDAkQo1C0p)) | The causal prefix-only adaptation is already in [Learning Round 1](../../campaign_2026-10-17/research_reset_r1/sources/LEARNING_ROUND1.md). First use the existing raw-data summaries to show that warm-up statistics predict later feature drift; full-sequence RevIN statistics would violate causality. |
| Filtered regime-dependent predictors | Hamilton's switching-regression model ([paper](https://doi.org/10.2307/1912559)) | Overlaps the state/regime proposals in [Microstructure Round 1](../../campaign_2026-10-17/research_reset_r1/sources/MICROSTRUCTURE_ROUND1.md). A useful new arm would need a causal filter, an input-only gate, and a matched complexity control. |
| Metric-aligned loss | Weighted prediction objectives | `wmse` is already the default R4 loss and uses the current metric's target-magnitude weighting. Re-deriving the score is worthwhile; spending on the same weighting is not a distinct experiment. |
| Predictive self-supervision | Contrastive Predictive Coding ([CPC](https://arxiv.org/abs/1807.03748)) | Masked input reconstruction is already proposed in [Learning Round 1](../../campaign_2026-10-17/research_reset_r1/sources/LEARNING_ROUND1.md). Input prediction alone is not a score result; a paired target probe and an ancestry-clean student comparison would be required before scaling it. |

These proposals remain eligible as descendants when they add a different
mechanism and a control that changes the decision. The overlap review is a
portfolio ordering decision, not a claim that the underlying families are
false.

## Research allocation rule

For each GPU job, write down before launch:

- the decision it can change and which teacher, student, or teacher/student
  relationship it belongs to;
- the exact strongest matched control, fitting ancestry, input/label timing,
  metric, checkpoint and component membership;
- the full resource envelope and what artifacts survive interruption;
- the practical success case, an uncertain case and the recipe-specific
  negative case; and
- the next branch of work for each of those outcomes.

This keeps the campaign moving without requiring an early score to beat the
current system. The immediate use of compute is to finish R4 cleanly, then spend
follow-up GPU time on the result's most informative interaction with a teacher,
student or complete endpoint. The public submission comes after local ranking
and serves only as a transfer check.

## Provenance and evidence limits

- Based on the user-directed local geometry campaign, R1–R3 records, the
  independent read-only reframe proposal, repository overlap checks and the
  independent R4 runner audit.
- Literature links support analogies only; they provide no Wunder score prior.
- All numerical local results cited above are exposed, small-panel evidence.
- No new fit, submission or public score is claimed by this note.
