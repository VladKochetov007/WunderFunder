# Independent target-generator falsifier: translated winning references and swing boundaries

Proposer `/root/architecture_proposals_0930`, 2026-09-30. Design only; no fits, probes, jobs, GPU or live model edits. Read the frozen current-code/full112 reports and E-162 plus its timing report. E-187/188/248 were consulted only to resolve the old episode-count wording. This proposal is ready for independent criticism before an actor implements it. The proposer does not authorize promotion.

## Correct the premise before testing a generator

The old approximately30 number is **weighted effective episodes**, not evidence that a sequence has only30 raw target changes. E-187 reports median28.66 effective episodes on its inspected validation sample. E-188 reports sampled TRAIN class3 effective episodes117.99 under another weight definition. E-248 aggregates over61k scored episodes per fold. None supports substituting30 for the raw number of target plateaus.

A 20,000-row sequence can have many target changes at quote events yet only a few long plateaus carry most weighted row mass. The new reports find76,815 target-change rows on640,000 held-out TRAIN rows, with only3 changes when the price code is unchanged. That identity is compatible with several generators that reevaluate only at publications; it does not distinguish fixed event horizon, extrema, stopping rules or stale label serialization. A generator interpretation must count raw changes, reference changes and weighted effective episodes separately.

The first proposed cheap exclusion therefore needs **no transition decoder**: on the new frozen TRAIN panel count source-float32 target-pair plateaus, price-code plateaus, a6 endpoint events and their intersections. Report unweighted episode count, rows/episode, and `N_eff=(sum_j w_j)^2/sum_j w_j^2` with `w_j=sum_rows_in_episode (abs(clip(t0,2))+abs(clip(t1,2)))`, using supplied target units and no scoring masks. This row-mass diagnostic is not an independent-outcome count or a metric estimate. If raw target/reference changes are numerous, reject any theory whose argument relies solely on about30 raw episodes.

## New question and fixed partial hypothesis

**Question:** after removing current-quote displacement in the accepted supervised coordinate, does the **winning target reference** equal the next globally segmented swing extremum, with reference changes aligned to extrema rather than arbitrary origin-anchored stop hits?

**Primary hypothesis H-SW2:** a deterministic publication-path process defines peaks/troughs by a2-decoded-lattice-unit reversal from a running extremum. At each origin, the future winning reference is the nearest-integer midpoint of the next such extremum. The extremum is backdated to where the peak/trough occurred and confirmed later. This could create a reference shared across a basin of origins and tier changes related to the next trend run. It is a **partial winning-reference mechanism**, not a complete explanation of the losing tail or a proven physical target.

This differs from the failed first-adverse and barrier probes: their threshold was relative to the origin and primarily explained losing offset k. Here the threshold is relative to the evolving peak/trough of an input path, the segmentation is global within a fully known block, no target chooses the direction/extremum, and the primary outcome is the winning reference rather than k. No k=winner rule, tail-law grid, first-adverse stop or inverse-clock fit is reopened.

The prior is low. E-162 found winning side approximately first-move sign on class3 and about85.3% exact agreement with the rounded next-four-midpoint mean on its inspected sample. That is evidence for local future-event structure, not long swing extrema. The new law is useful chiefly because a small number of high-confidence counterexamples can reject it without training.

## Frozen data contract

- Provided TRAIN parquet only. First choose64 fit groups and32 test groups from distinct test chain components by the fixed key `SHA256('target-swing-v1|'+group_identifier)`. Exclude all earlier reconstruction/provided-decode/clock study components and chain-linked fit groups using split metadata; freeze exact group IDs and source hashes **before reading new targets**. Do not stratify selection by held-out targets. If fresh components are insufficient, report blocked/incomplete rather than silently reuse the inspected96-group panel as independent confirmation.
- Fit accepted code-spread and local transition tables only on fit TRAIN groups, interior pairs only, reproducing the already frozen map definitions: current spread support>=3/confidence>=.95 and transition support>=3/confidence>=.80. Fit target lattice conventions within fit TRAIN, retain their exact identity. No held-out target differences are used to expand/repair the movement table.
- Holdout target labels are read by the grader for diagnosis after frozen decoder/segmentation predictions exist. No validation, `seq_ix`, prediction/scoring masks, external data or external numeric transformations. Group/component IDs select splits only.
- Process the full20,000-row timeline of every selected group; future input is permissible **for generator diagnosis only**. No generated mark is a causal inference feature or a candidate submission.
- One CPU thread, hard2GiB RAM, shared `wnn-campaign.slice`, no GPU, each bounded pass RuntimeMax60s. Stop on a cap and preserve incomplete denominators. Expected streaming time10–40s for each decoder-fit/path-grade pass, estimated from prior small report passes, not measured. No parameter search.

## Construct coordinates while exposing circularity

For a known contiguous block anchor current bid at `B_0=0` and ask at `A_0=s_current`. Integrate accepted local displacement votes `delta A=-delta t0` and `delta B=+delta t1` from the **fit** table. These are decoded lattice coordinates, not identified physical bid/ask. Define `M=(A+B)/2`.

For diagnostic interior rows with a valid code spread and `t0+t1=-s_current`, translated side references are

```text
r0=t0+A
r1=t1-B
F0=r0; F1=-r1=B-t1.
```

A common mark would give F0=F1. Interior-to-interior delta votes were trained from target differences under exactly this assumption. Constant translated references on those transitions are therefore **decoder-imposed/decoder-consistency evidence**, even if edge values transfer to held-out components. Do not count them as independent confirmation of a mark process, and do not declare an absolute price potential from a current-code loop.

The novel primary test uses **outside rows** with a strictly positive winner, whose target values never supplied the interior-only displacement votes:

```text
if t0>t1 and t0>0: F_win=A+t0
if t1>t0 and t1>0: F_win=B-t1
```

All outside rows remain eligible outcomes once the required input prefix/future path is known; branch choice is used only to grade the observed winning target. The generator prediction selects the next input-defined extremum without consulting this branch. Touch rows with max target0, interior rows and branch-switch transitions form prespecified secondary strata. Never use held-out winning sign to choose a matching future peak versus trough.

Outside testing removes direct reuse of its label difference in map fitting, but the decoder still uses target-derived coordinates. Thus even fresh component success establishes only a law **conditional on this supervised coordinate**. Stronger physical identification requires an independent feature-only displacement decoder or independently observed supplied quote coordinates, neither currently exists. Report this gap instead of hiding it behind holdout exactness.

### Unknown prefixes and path validity

Any unknown code transition, unmapped current-spread anchor, or a6 endpoint with unchanged code and unidentified displacement breaks the known block. No unknown edge is set to zero, inferred from its held-out target, or skipped. Later blocks may establish a new arbitrary translation gauge, but no reference-continuity or horizon claim crosses the break. Count left-censored prefixes, unknown future prefixes, no-confirmation-within-horizon and sequence-end truncation separately.

Decoded nonzero-event equivalence with a6 is valid only on accepted mapped transitions. Origins are initialization, not quote moves. Known zero transitions can advance publication rows while holding A/B. When a known nonzero transition has zero midpoint motion through a spread change, it still consumes an endpoint event; do not discard it from the horizon clock. The three previously reported quiet endpoint exceptions remain unknown, not false zero moves.

Before grading marks check `A-B=s_current` wherever the independently fitted current-code lookup is known. Failure invalidates that block for this conditional coordinate claim and is recorded as decoder inconsistency, not a generator mismatch. This consistency is also target-supervised and not independent physical proof.

## Fixed swing algorithm and competing explanations

Use decoded M, thresholdD=2, no smoothing and no time warp. Initialize direction unknown at each known block. Establish a direction only after an excursion of magnitudeD from the initial M; censor origins until the **first subsequent reversal confirmation**, avoiding arbitrary initial-direction artifacts. In up mode keep the earliest running maximum P. When `P-M>=D`, emit the peak at its original index e, with confirmation index c, then enter down mode with trough initialized at the confirming M. In down mode reverse symmetrically at `M-Q>=D`. Update an extremum on a strict improvement only; ties keep the earliest index. Fixed threshold2 is one lattice unit convention of the supplied decoder, not seconds, money or a universal physical tick.

For each uncensored origin t, pick the earliest emitted extremum with e>t and confirmation c no more than64 mapped nonzero endpoint events after t. Require every intervening representation transition known, including zero transitions. Predicted winning reference is `F_hat=B_t+floor(M_e-B_t+0.5)`: nearest lattice relative to the current decoded bid, with a fixed half-up tie rule. This form is invariant to an arbitrary common translation of A/B, avoiding an unidentified absolute rounding origin or banker's parity. No alternate rounding is selected; E-162's rare-tie evidence makes rounding a convention, not a gain mechanism. Compare the observed winning **relative** displacement `F_win-M_t` with `F_hat-M_t`. Origins at/after a peak use the next extremum; no past extremum is substituted to obtain a match.

Prespecified controls, computed on the identical eligible origin set:

1. Current midpoint rounded relative to current decoded bid under the same half-up rule.
2. Midpoint after the first subsequent **nonzero midpoint** move rounded relative to the origin bid, without using the target sign. State zero-midpoint-event coverage explicitly.
3. Average of next4 mapped nonzero endpoint-event midpoints rounded relative to the origin bid under the same rule. This is the old local-horizon family as a background negative/partial control, not a new winning claim or selected horizon.

The primary swing rule must outperform the strongest frozen control by at least3 percentage points in exact outside winning-reference accuracy and attain>=99.5% exactness on the high-confidence stratum to remain a plausible nearly deterministic rule. The thresholds are continuation criteria, not statements that a99.5% approximation recovers the exact generator. A single reproducible valid counterexample falsifies mathematical exactness; three counterexamples on different fresh components with high-confidence paths reject the substantive rule rather than blame one decoder alias.

Use the original map gate as full primary coverage and separately predeclare high-confidence paths where every used transition has fit support>=20/confidence1.0 and anchors meet the same stricter gate. Never tighten support after observing errors. At least8 fresh test components with eligible outside origins are required for continuation. If coverage cannot meet this, the result is inconclusive, not a passed law. Report class strata only retrospectively; no class-specific threshold/horizon is fitted.

## Changepoint and cheap exclusion outputs

Emit counts/denominators, not just correlations:

- Raw target-pair plateau count; price-code and a6 event count; translated winning-reference plateau count; the weighted effective count for each stated weighting.
- Exact mark matches, absolute/relative errors, eligible-origin coverage by component and diagnostic class, unknown/truncated/no-confirmation mass in the complete denominators. Also count distinct tested edges and components, because repeated rows are dependent.
- Input-defined extremum timestamps e and confirmation timestamps c. Report winning-reference changepoint alignment separately to e versus c on complete known blocks. **Backdated e** supports future-extremum assignment; **confirmation c** would suggest online hold/latch; neither alignment may use interior-imposed invariance as its primary denominator.
- Full and high-confidence outside rows, touch rows, interior rows, branch switches and code-quiet exceptions separately. Changes in target pair at a quote move need not equal changes in translated future reference.
- Minimal immutable counterexamples: group/component, source row indices, code keys and support/confidence, mapped A/B/M trajectory, observed targets, candidate e/c, unknown flags and exact identities. These are diagnostic TRAIN excerpts, not model training artifacts imported from outside.

Immediate cheap exclusions:

1. **Few raw label episodes:** reject if the raw count differs substantially from the purported30; do not turn effective count into raw sparsity.
2. **One common future mark explains both targets:** already contradicted by reported outside-tail identities. Recompute `t0+t1+s_current` without path integration; common-mark models require0 on all rows, which outside data need not satisfy. Do not select extrema to fix this algebraic failure. Winning-reference success is partial only.
3. **Nonzero closed-code displacement implies static absolute code price:** already false in the current report. Preserve the gauge/integrated-path construction, not a static price lookup.
4. **A6 means every physical quote move / no endpoint means zero:** unsupported on unknown transitions; report those as unknown rather than extend the mapped identity.
5. **Held label between all quote moves implies sparse generator:** fails to distinguish publication-driven recomputation from genuinely shared future marks. Require translated reference changepoints outside the tautological interior stratum.
6. **Next swing predicts the mark:** reject on the declared accuracy/control/component criteria, or preserve high-confidence counterexamples. Do not changeD, horizon64, rounding, branch or selected peak after seeing them.
7. **Swing law predicts losing offset:** not proposed. Previously failed conventional/doubled first-adverse and k-winner laws remain failed. Even successful winning marks cannot resolve their tail failure or justify a full outcome head.

## What a result would authorize

Failure cheaply closes H-SW2 and records whether the cause was decoder coverage, conditional coordinate mismatch or mark-rule counterexample. A pass permits independent reproduction and an independent-input-decoder search. It does **not** authorize auxiliary labels, model fit, inference memory, packaging, or promotion. A full generator requires separately reproducing both targets including tail copula and publication update policy. No public-score effect is estimated from this reverse-engineering diagnostic.

Possible causal model consequence only after that separate proof: a state model could forecast the conditional next swing/reference regime or its end, using past decoded event summaries and explicit unknowns. The future emitted peak and retrospective target references themselves are never usable at live inference. Source identity, original units and physical time remain unknown.
