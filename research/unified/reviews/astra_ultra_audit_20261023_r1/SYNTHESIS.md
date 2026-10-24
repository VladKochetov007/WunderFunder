---
kind: review
status: complete
created_utc: 2026-10-23T17:17:50Z
updated_utc: 2026-10-23T17:28:29Z
hypothesis_ids: []
experiment_ids: [E-ASTRA-ULTRA-AUDIT-20261023-R1]
evidence_paths:
  - research/unified/reviews/astra_ultra_audit_20261023_r1/MANIFEST.json
  - research/unified/reviews/astra_ultra_audit_20261023_r1/EVIDENCE_MAP.md
  - research/unified/reviews/astra_ultra_audit_20261023_r1/RECONCILIATION.md
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/USER_STOP_R1.json
---

# What our research may be missing, and which ideas may be misleading

## Main judgment

We have credible opportunities, but several explanations have moved ahead of
their evidence. The largest immediate opportunity is to identify **what useful
correction an idea contributes to a strong complete endpoint**, then distinguish
that correction from cheaper alternatives. This can preserve ambitious research
while preventing another expensive comparison that answers the wrong question.

The review finds six especially consequential gaps:

1. The history model improves a fixed blend while losing alone, but ordinary
   second-seed diversity has not been priced against it. History-specific
   information, optimization changes and nonlinear recalibration remain live
   explanations.
2. Small residual heads learn on in-sample parent predictions and are evaluated
   on parent-excluded groups. This may make the head a poor instrument for
   deciding whether a feature is useful, even when evaluation ancestry is clean.
3. A continuing online estimator can beat a short frozen warmup simply because
   it is more mature. That would be useful, but would not identify nonstationarity
   or successful tracking of a changing relationship.
4. Initial prediction parity is valuable but does not equalize learning
   geometry. Several proposed controls preserve the information they are meant
   to remove or compare different optimization exposures.
5. Distillation can protect an inherited function. Its value is not measured
   solely by improvement over an already-distilled initializer. Conversely,
   a stronger standalone teacher need not add useful information to a student
   that already carries its signal.
6. Research cost includes transport, full-sequence state reconstruction, exports,
   grading and native serving. Completed fits without interpretable endpoints
   have repeatedly consumed resources without resolving the scientific question.

There are also concrete corrections. The exact H512 local/B300 score replication
**succeeded**; an earlier contrary interpretation compared a single cell with a
multi-fold mean. Warm-history initial parity **is complete**, although its paired
quality comparison is not. The projected checkpoint mean **was already graded**;
one independent reviewer missed the newer receipt. Repeating that work is removed
from the recommendations. A proposed fixed flag derangement is also removed as
an identifying control for an unrestricted dense head. [Correction record](RECONCILIATION.md)

These findings do not establish that history, age, online features, large teachers
or alternative losses are bad ideas. Most need a sharper experiment; some exact
recipes already have adverse evidence; some only failed to finish. The >0.700
and top-1 objectives remain objectives. No new model, score, public forecast or
submission is produced by this review.

## What was done, and what this evidence means

Ten fresh-context `gpt-6-astra` agents at `ultra` reasoning effort completed
separate reports, in batches of at most three active subordinate roles. Each
received a different lens, the existing source navigation, the same evidence
standards and a zero-execution contract. No recursive agents or external CLI
model sessions were used. Root read every report before this comparative
synthesis and checked selected claims against original text/JSON receipts.

The outputs contain **40 hypothesis cards and 70 decision-changing questions**.
Thirty-nine cards propose unexecuted comparisons; card 09-D unknowingly repeats
a completed comparison. Several cards are alternative formulations of one
question, not additional experiments. All are mapped in the [idea register](IDEA_REGISTER.md);
the [question register](QUESTION_REGISTER.md) preserves all seventy questions.

| Review | Its main contribution | Where it challenges an attractive story |
|---|---|---|
| [01: evidence](01_evidence_and_selection.md) | Matched endpoints, ordinary-seed control, component influence | A small positive blend delta does not identify history; an unfinished panel is not a negative result. |
| [02: representation](02_representation_and_hidden_structure.md) | Age/position/censoring, numerical atoms, restricted correspondence tests | Anonymous encodings and visible patterns do not establish field semantics or useful invariance. |
| [03: memory](03_memory_and_time.md) | Separates carried state, credit horizon and readout access | Longer forward memory, longer gradients and better use of existing state are different interventions. |
| [04: objectives](04_objectives_and_metric.md) | Effective row weights, actual loss dispatch, nonlinear calibration | “Pearson needs a new loss” and “clipping suppresses Adam steps proportionally” are inadequate diagnoses. |
| [05: scaling](05_teacher_scaling_and_capacity.md) | Capacity/exposure tradeoffs, credit and unique-data anchors | A larger model cannot diagnose undertraining, but simple-baseline literature cannot rule out scale. |
| [06: endpoints](06_ensembles_and_distillation.md) | Conditional ensemble value, KD retention, fit-side residual mismatch | Standalone model rank and incremental endpoint value are different; evaluation exclusion alone is insufficient. |
| [07: identification](07_causal_identification_and_controls.md) | Mature stationary controls, learning geometry, age identity | A control must remove the intended mechanism while stating what else changes. |
| [08: transfer](08_transfer_and_information_limits.md) | Hidden selection, target maturity and existing stress panels | Input reconstruction does not produce target feedback, a physical clock or a performance ceiling. |
| [09: delivery](09_research_value_and_delivery.md) | Remaining decision value, duplicated controls, whole-sequence exports | Cheap heads and zero-fit analyses can still be expensive; its projected-mean recommendation required correction. |
| [10: red team](10_unrestricted_red_team.md) | Bounded response-speed test, correction decomposition, duplicate-family check | Failed smoothing does not imply more memory; a sophisticated head can beat a broken control and lose to the parent. |

Independence here means independently formed proposals, not ten statistically
independent measurements. The reviewers share a model family and repository.
Reviewer 10 incidentally saw the title/description of the new source register in
a common index, disclosed that exposure, and did not open it or any sibling
report. Root preserves that qualification in the manifest. No private reasoning
transcripts are stored.

This is a tier-C source/literature review. No fits, scientific evaluations,
imports/tests, profiles, prediction-array reads, exports or provider jobs ran.
Existing evaluation folds remain exposed. Conditional component bootstrap
intervals describe uncertainty in fixed predictions; they do not measure
training-seed variability or public transfer. Native agent token/cost accounting
was not supplied. The experiment stop remains effective.

## The evidence we should carry forward accurately

The [evidence map](EVIDENCE_MAP.md) contains twenty-six source-bound entries.
These are historical observations read from receipts, not recomputed results.

| Question | What the record says | Appropriate interpretation |
|---|---|---|
| Does current i1 matter? | Standard B beats masked-i1 A by +0.01590144 on F0/1001 and +0.01110844 on F2/1002. | Strong evidence for i1 in this small cold recipe, not proof of a particular cross-instrument interpretation. |
| Does extra history help? | C loses to B by −0.00353711 / −0.00390342. Fixed 90%B/10%C gains +0.00035953 / +0.00027145. | Useful conditional blend signal in two exposed cells; standalone and endpoint rankings disagree. |
| Is that gain precise? | B/C component intervals are [−0.00001187,+0.00070893] and [+0.00008223,+0.00043398]. | Positive but small; one interval crosses zero. No automatic rejection, significance claim or public extrapolation. |
| Is history the only route? | 90%B/10%A gains −0.00002308 / +0.00054446 despite A removing i1. | A different information restriction can also supply useful errors. Ordinary diversity is a serious comparator. |
| Did warm history pass initial parity? | Selected full-sequence receipts report zero output error for both extensions on F0/F1. Only one control fit/export landed. | Initial parity is resolved on that scope; the scientific warm comparison remains incomplete. |
| Did H512 fail to reproduce B300? | Exact F0/1003/CP5 scores differ by about −1.27e−9. | The score reproduced. This does not identify an architecture effect or improve the submission. |
| Does H512 improve the retained endpoint? | Same-host replacement loses −0.007362916; host-reference drift is about −0.000000588. Sampled FP32 recovery is tiny. | Runtime/precision alone are poor explanations on inspected scopes. Exposure, ancestry and endpoint composition remain different. |
| What transferred publicly? | Reported TL3 0.66832; two-parent 0.66771; H512 replacement 0.66667. Two-parent saves 641 reported seconds. | Dated user-reported outcomes, not freshly verified portal/payload bindings. Retain a quality/runtime frontier, without tuning from these scores. |
| Is NLL better? | Matched NLL−WMSE −0.0000436109; interval [−0.0006230104,+0.0004552434]. Fixed-blend difference −0.0000038521. | No demonstrated advantage or equivalence. One exposed continuation with important optimizer questions. |
| Is KD exhausted? | Alpha0−alpha0.9 reaches −0.01431282 at primary CP5; interval [−0.03116723,−0.00112208]. | Retain this supported warm KD recipe. One historical FP16 seed and inherited distillation limit broader conclusions. |
| What did B300 establish? | Twelve width/depth pairs favor wide by 0.00077732 on the equal-fold mean; geometry partials favor baseline; age/innovation have no scored pairs. | A recipe prior, adverse evidence for exact geometry bundles, and unresolved age/innovation. No scale law. |
| What did online/age establish? | Seven of twelve online outputs saved, no grade. Age CPU profile failed before data reconstruction, zero fits. | Delivery failures, not scientific negatives. Saved F0 online contrasts may be recoverable under a new declared analysis. |
| What did checkpoint averaging establish? | Fixed oldZ4/mean versus oldZ4/WMSE adds +0.0000109615; interval [−0.0000830101,+0.0001132356]. | Already evaluated, tiny and uncertain. Do not redo the same grade because a summary is stale. |
| Can a new head beat its control and still fail? | E421: untouched 0.682090, joint CE 0.630341, factorized CE 0.593983, new WMSE control 0.492579. | Yes. Preserve the untouched competent parent in every residual/readout screen. |

Absolute scores in different rows are not a common leaderboard. Training
membership, source, target treatment, checkpoint, precision, sampling and endpoint
composition differ. Exact numerical comparisons belong to their linked packets.

## The most valuable reframing: ask what the correction contains

### History is a promising component with an unresolved explanation

The strongest recent history evidence is the fixed B/C mixture, not C's standalone
score. Rejecting C for losing alone would discard an observed endpoint benefit.
Calling the mixture a demonstration that longer memory is needed would also
overstate the result.

At least five explanations remain:

- C supplies target information missing from B's representation.
- C learns a different function because its random history columns are active
  from initialization.
- Ordinary fitting randomness supplies similarly useful diversity.
- C approximates a nonlinear recalibration of B's output.
- The benefit depends on the particular component/activity composition.

Clipping alone is already an inadequate explanation: F0 gain is +0.00036506
without prediction clipping and +0.00035953 under official clipping. Likewise,
positive covariance after an affine projection does not exclude a nonlinear
function of B. These distinctions matter because the cheapest useful endpoint
could be a scalar map, an ordinary ensemble, a history-aware student, or an
explicit memory branch.

The next question is not “is history good?” It is: **which information or
optimization change buys a better fixed endpoint per complete training and
serving cost?** Cards 01-A/06-A/09-B/10-C examine ordinary diversity; 07-A adds a
current-input nonlinear expansion; 04-C tests scalar sufficiency; 08-A and 01-D
test concentration. They should become a small chosen sequence, not six panels.

There is a real design disagreement to preserve. Reviewer 01 favors repairing
initial-function equality; reviewer 10 favors four new ordinary-seed fits,
two per original fold, reusing the exact historical B/C endpoints to compare
allocation value. The first asks a cleaner mechanism question. The second asks
a cheaper purchase question, with remaining stochastic confounds. Neither can
silently borrow the other's controls. The already planned warm-history panel
is a third, strong-parent question; its completed control is reusable only for
its exact frozen counterpart.

### Residual learning may be testing the wrong residual distribution

The online and age heads correct parents that fitted the head-training groups.
They then face parent-excluded evaluation groups. An in-sample parent can leave
small, atypical errors, so its head may learn a correction unlike the one needed
on unseen components. Clean evaluation ancestry does not remove this mismatch.

This does not invalidate matched feature comparisons automatically. Both arms
may share it. But it can interact with a feature: age or history might explain
out-of-sample mistakes that barely exist on the fitting side. A negative head
screen could therefore be a false negative for an end-to-end representation.

Card 06-B proposes a parent-excluded head-training comparison, with a separate
evaluation set and explicit recursive ancestry. It should preserve a competent
untouched parent, matched fitting counts, a common correction architecture and
the same update budget. Different calibration populations remain a confound.
Full cross-fitting of new parents is not assumed free; the card uses existing
exclusion opportunities only when source membership proves them.

A related information gap is the head's conditioning. Current small residual
inputs do not include the parent's hidden state or parent prediction as a head
feature; the parent prediction is added afterward. Thus a useful feature may
need state-dependent interpretation that the small correction cannot express.
Card 03-B separates raw-input readout, hidden-state readout and history access.
This is worthwhile only after counting state export and storage costs.

### Useful disagreement should be examined in the complete endpoint

For a base p and correction d, low correlation between models does not suffice.
Away from output clipping, the correlation derivative depends on both target
covariance with d and d's covariance with p. A model can be diverse in useless
directions, or weaker overall but helpful in the base's residual direction.
Official clipping makes direct fixed-endpoint grading indispensable.

This suggests inexpensive questions after resume: is useful correction mainly
a persistent offset, a recent change, a target-specific component, or concentrated
in a few source components? Card 10-B fixes one slow/fast split of C−B rather
than searching many filters. Card 04-C fits a bounded scalar comparator rather
than asserting that nonlinear recalibration must explain the gain. These analyses
can identify a compression path, but must not become unrestricted tuning on
already exposed labels.

## Time, age and adaptation: separate mechanisms that currently share a name

### State persistence, credit assignment and readout access

A carried recurrent state can summarize observations long before the latest
training chunk. A short detach horizon limits gradient credit through those
earlier operations; it does not erase the forward state. A head may also fail
to extract information already stored. Increasing context length, increasing
gradient horizon and expanding the readout solve different possible bottlenecks.

The S14 packet already proposes 500-row microchunks and a 2,000-row optimizer
window with detach at 500 versus 2,000. Both historical forty-update FP16 profiles
completed; the quality fits did not. Cards 03-D/05-B correctly revive this as a
specific unanswered question. A strict-FP32 descendant needs new cost evidence,
but not a new story that nobody previously considered temporal credit.

Card 03-A's finite-context replay can reveal sensitivity to older state. Its
state resets are interventions outside the fitted trajectory, so a loss after
reset does not directly measure the optimal training context. Preserve feature
clocks while changing only the intended recurrent state. Card 03-B asks whether
existing hidden state makes explicit history redundant. Together they provide
better diagnoses than choosing a long-context architecture solely by intuition.

### Row time and event time can both matter

Selected-price inactivity is not full-input inactivity. The narrow census found
1,253 eligible target changes outside the earlier selected-i0 conjunction,
including 92 scored-valid exceptions. A separate full-112-feature sample found
no adjacent identical full vectors across 256 sampled sequences. Neither
supports a universal “nothing happened, skip the GRU” rule.

An inference-only skip rule changes a row-trained recurrence and can lose timing,
small innovations or readout updates. It remains a possible efficiency
intervention, but its causal state semantics and complete native cost need their
own evidence. Literature on learned skipping does not validate a retrofitted
novelty threshold.

Cards 02-B/03-C offer a more targeted representation: retain the original row
clock and raw path while adding a change-count summary with a duration escape
path. Event time can compact repeated conditions; elapsed row time may itself
predict the target. Do not enforce invariance until a matched comparator earns it.

### Age may represent several different things

A field-age counter can encode time since a change, sequence position, activity,
censoring at sequence start, saturation, or feature support. Before its first
observed reset, an age counter is a lower-bound observation of an interval that
began outside the window. A seen-change flag can distinguish that case, though
it may be redundant with uncapped age plus exact position.

The current grouped-age versus individual-age screen is not a pure “more clocks”
experiment. Under common reset/cap conventions, a group counter tracks the
minimum field age. Repeating seven group ages and supplying 112 individual ages
also changes rank and learning geometry. Zero-output initialization equalizes
the starting predictor, not those properties.

Cards 02-A, 07-C and 08-D separate age beyond activity/position, identity beyond
the within-group age distribution, and censored versus observed intervals. The
08-D fixed derangement is inadequate with an unrestricted dense head: an
invertible column permutation does not remove information. Restricted shared
encoders or pooled/sorted summaries ask different, better specified questions,
with their own inductive-bias confounds.

### Continuing estimation is not yet evidence of drift adaptation

The online warmup uses only 83 updates in the cited short-prefix case, with
variance decay 2^(−1/512). The initialized variance term retains coefficient
approximately 0.894. This is not the fraction of total variance, but it makes
estimator maturity a concrete alternative to nonstationarity.

A continuing estimator can gain from more observations under a stationary law.
A frozen short-prefix estimator can lose because of initialization, coefficient
noise or unequal intercept updates. Before interpreting a B/C difference as
tracking drift, compare equally mature recent and older windows, as in card
07-B. A useful stationary estimate would still be valuable; the name changes
the justified next experiment.

The current twelve-head panel has no quality result. Its six F0 outputs cover
all A/B/C and both lags, while F1 is incomplete. A future fixed F0 exploratory
grade could inform whether missing-cell recovery is worthwhile. It must not
retroactively satisfy the whole-panel gate. Lag-independent A controls have
matching reported prediction hashes in F0; future exact reuse can reduce cost
only after source, membership and coverage checks.

## Objectives: inspect the effective measure before inventing a new loss

For one target, set Z=clip(Y,−2,2), use scored mask M, and tilt the observation
distribution by M|Z|. The optimal unrestricted weighted squared-error predictor is

`m(X) = E[M|Z|Z | X] / E[M|Z| | X]`.

Under finite nonzero variance, its positive affine direction also maximizes
population correlation under the same tilted measure. Since Z is bounded, m
is itself an admissible bounded prediction. This derivation is already in the
project; it is not a new result from this audit. [Metric and objective review](04_objectives_and_metric.md)

The practical question is why the implemented finite model departs from that
ideal. Propensity powers, masks, chunk normalization, target aggregation,
teacher mixtures, auxiliary objectives and variance weighting can all alter the
effective learning problem. Equal official weight for two target correlations
does not imply equal finite-capacity training emphasis when one WMSE denominator
pools their residual weights. Chunk-normalized gradients need not equal the
gradient of one globally normalized risk.

There is a concrete source trap: current `main_loss` handles teacher caches in a
hard-coded WMSE branch before generic objective and running-normalizer dispatch,
even at alpha zero. A future flag-only loss comparison could test no change.
The historical NLL experiment used a different pinned producer that did implement
NLL. Verify the executed branch; do not use present source to rewrite its history.

NLL clipped every update while WMSE clipped roughly seven percent. That does
not mean its recurrent Adam updates were four percent as large. Adam's
normalization, time variation, epsilon, auxiliary directions and weight decay
matter. Also, a target with 90% teacher and 10% observed label can have much
smaller conditional label-noise variance than the original label. Learned
variance is not automatically market uncertainty.

The useful next diagnostic is an effective-weight and actual parameter-step
fingerprint at a state with learned variance, not only an initial zero-variance
gradient. Cards 04-A/04-B isolate chunk normalization and variance weighting.
They remain proposals; the broader variance-head sweep has no demonstrated
quality advantage.

Finally, distributional targets deserve an explicit optimization rationale.
E421 already shows why beating a newly degraded WMSE head is insufficient.
A soft-histogram descendant in 10-D is retained at low priority only if it differs
materially from prior executed recipes and preserves the original competent
parent. A literature citation alone should not reset a failed lineage's history.

## Representation and information limits: useful uncertainty, not a ceiling

Anonymous feature values invite semantic stories. Repeated atoms might mark
states, discretization or encoding artifacts. Apparent cross-instrument pairs
might reflect meaningful correspondence, or a numerically convenient layout.
Sequence moment arcs can arise from mixtures and occupancy without identifying
two physical regimes.

These are opportunities for restricted, falsifiable models. Cards 02-C/08-C
separate exact atom identity from occupancy and a generic nonlinear numerical
basis. Dictionary statistics must be fitted only on fitting groups and evaluated
on unknown atoms and different components. Card 02-D tests a shared paired
encoder with real versus deranged pairing while retaining raw information.
A fixed permutation before a fully flexible dense layer would not test that
prior; the restricted architecture is essential to the question.

Retrospective target classes are explanatory partitions, not causal serving
features. Class 3 accounts for 47.69% of combined magnitude weight in the cited
EDA; that is not 47.69% of the pooled score and does not identify a reachable
quality ceiling. Its long selected-clock ages motivate checking state and timing,
but do not establish target maturity.

The target-feedback contract is particularly important. Predicting an observable
future input gives a valid delayed input-based learning signal once that input
arrives. It does not mean the competition target becomes observable at the same
lag. Observable-input adaptation remains eligible; target-based online updates
need independently documented target availability.

Selection adds another uncertainty. The scored-valid subset is only a fraction
of all valid rows, and the serving callback does not receive its mask. If selection
depends on outcome information beyond causal history, a propensity model of
selection given that history cannot automatically recover the selected target
law. This challenges a universal interpretation of propensity correction, not
its possible empirical usefulness. Card 08-B tests fixed weighting powers;
it does not pretend to solve missing identification assumptions.

No existing evidence establishes a Bayes ceiling, a universal target clock, or
that current models already exhaust the information. These unanswered questions
should guide tests without becoming excuses to abandon scale.

## Teacher scale and distillation: keep ambition, improve the comparison

The large-teacher proposal already distinguishes unique data, presentations,
updates, context and gradient horizon. Repeating these cautions as if they were
new would add little value. The review's concrete addition is a comparison that
can separate capacity from buying more training.

Card 05-C proposes compact teacher anchors around 19M and 75M parameters and
a small-long versus larger-short contrast, six total teacher fits. Such anchors
can establish whether a candidate architecture learns the task and how exposure
affects its slope. They are not a serving-size ceiling or a mandatory permission
gate before the user's larger teacher research. The 0.206/0.679/1.644B proposal
remains a separate prospective scale ladder, with no measured local scaling law.

Card 05-D compares unique component diversity with repeated presentations using
fresh, ancestry-restricted models. A small-data child initialized from a parent
that already saw all data would not identify the value of unique data.
Card 05-A inventories saved learning curves first; evaluating every checkpoint
can cost sixteen forward passes and consume more selection exposure. Only
missing, decision-changing points deserve that expense.

The partial B300 wide/deep result gives a pragmatic recipe prior. It changes
width and depth together, and fold behavior is heterogeneous. It does not
establish that parameter count is the bottleneck, or that deep models cannot
win with another optimization recipe. Language-model scaling and time-series
papers offer design components, not numerical laws for these correlated rows.

Distillation has two separable purposes: retain a strong inherited predictor,
and transmit information the student does not yet have. The matched warm alpha
contrast supports retaining current KD. A small gain over the initializer
does not negate its protection against harmful continuation.

For new transfer, compare students trained against a retention anchor versus
an explicitly complementary teacher contribution, as in 06-C. Count student
fits, teacher costs and cache production separately. Teacher quality can also
be a legitimate research objective in its own right under the teacher-first
instruction; immediate successful compression is not a universal promotion gate.
A later competition endpoint still needs measured complete serving feasibility.

## A proposed order of questions after an explicit resume

Nothing below launches work or changes the stop. The ordering is a root judgment
about information value, not a vote count or an estimated numerical reward.
Do not execute the entire list. Start with one bounded question whose two possible
answers would change a concrete decision.

| Priority | Proposed discriminator | New fitting count | What would change |
|---|---|---:|---|
| 1 | Freeze an exploratory F0 grade of all six saved online heads, both lag contrasts | 0 | Decide whether recovering missing unique F1 heads deserves spending. Artifact failure remains delivery evidence. |
| 2 | Apply existing fixed activity/i1-frequency/support panels to B, C and their fixed mixture; include component influence | 0 | Distinguish broadly distributed contribution from composition-sensitive gain; no new blend weights. |
| 3 | One bounded causal response-speed probe, five fixed transforms in the two cached B cells | 0 | Test a cheap output-response explanation before a new temporal architecture; retain an exploration reserve. |
| 4 | Ordinary diversity versus the fixed historical history mixture | 4 recurrent fits in 10-C's version | Decide whether a new history path earns cost beyond ordinary seeds; average predeclared seeds, never pick the best. |
| 5 | Complete the frozen function-preserving warm-history question if whole-path cost is feasible | 3 remaining of 4 planned fits; 1 exact completed control | Test stronger-parent transfer under the existing recipe; do not merge its estimand with priority 4. |
| 6 | Parent-excluded versus in-sample residual-head fitting | 4 heads if suitable exact parents/caches exist | Determine whether this head pipeline is a credible filter for age/online features. |
| 7 | Equally mature old versus recent input-relationship estimation | 4 heads | Separate useful estimation from drift adaptation, conditional on an online utility signal. |
| 8 | Temporal-credit contrast, preserving optimizer window and data order | 4 recurrent fits | Decide whether more gradient access merits priority over more state/capacity. |
| 9 | One age identity/activity/censoring comparison | 4–6 heads under a frozen selected design | Identify what the feature adds; correct the invalid dense permutation control first. |
| 10 | Compact capacity/exposure triangle or fresh unique-data contrast | 6 teacher fits or 4 fresh backbones | Allocate a separately authorized teacher-scaling budget from measured quality/cost rather than an assumed law. |

Priorities 1–3 are not free. They need exact payload availability, sequence
ordering, masks, ancestry and a feasible venue. No payloads were opened here.
The response-speed test is deliberately less established than online/history
recovery; its place protects a modest independent exploration path rather than
claiming it is the likeliest winner.

Scalar recalibration, slow/fast correction decomposition and effective-weight
diagnostics are useful alternatives if they answer the immediate allocation
question more cheaply. They should replace, not automatically augment, another
analysis. Four calibration jobs are still fits. A recurrent reset replay is
still inference. No-fit terminology must not conceal local load.

For each selected screen, write the exact unchanged endpoint, treatment,
matched controls, all new fits, all reused controls, primary comparison and
inclusive completion cost. Preserve raw deltas by fold/seed. Do not turn an
interval crossing zero into a universal stop rule, or small positive evidence
into an unconditional production claim.

## Which current ideas to retain, modify or deprioritize

| Direction | Disposition | Reason and reopening condition |
|---|---|---|
| i1 history | Retain and sharpen | Measured mixture benefit; missing seed/current-basis/calibration alternatives and strong-parent completion. |
| Per-field age/censoring | Retain unresolved | Execution failed before science; activity, position, identity and censoring need distinguishable controls. |
| Observable online relationships | Retain unresolved | Complete F0 subpanel exists; maturity versus recency remains unidentified. |
| Warm KD | Retain supported recipe | Matched removal harms the warm endpoint; new-information transfer is a separate hypothesis. |
| Large causal teachers | Retain as future research priority | No ceiling established; quality/exposure/context must be measured under separate compute authority. |
| Gradient-credit extension | Retain existing unanswered direction | Engineering profiles exist, quality comparison does not; current precision differs. |
| NLL/variance learning | Modify, diagnostic first | No measured advantage; effective targets and optimizer interactions remain unresolved. |
| Paired pooling / TensorSketch / atom encodings | Exploratory reserve | Plausible representation hypotheses, not established semantics; require competent raw-path controls. |
| Inference-only recurrent skipping | Modify strongly | Selected-price quietness is not full inactivity; row-trained dynamics and native path must be checked. |
| Output smoothing / early-label routing | Deprioritize exact tested recipes | Documented negative endpoints; does not close feature-side filters or observable-input adaptation. |
| Distributional readout reinvention | Defer exact repeats | H-111/E421 already address the estimand and supply adverse frozen-state evidence. |
| Tested geometry bundles | Deprioritize exact bundles | Partial matched evidence favors controls; materially changed descendants remain eligible. |
| Projected checkpoint mean regrading | Remove duplicate task | Already graded; tiny uncertain fixed-blend increment is the current evidence. |
| Public-guided coefficient search | Exclude as a selection method | Public receipts confirm transfer and diagnose disagreement; they do not provide a clean tuning set. |

## The questions worth keeping in active memory

The complete seventy-question archive is useful, but a compact active set should
prevent another proliferation of disconnected tasks:

1. What is the exact strong endpoint being improved, and what useful correction
   does the new component contribute to it?
2. Could an ordinary seed, scalar map or simpler current-input basis buy that
   contribution at lower complete cost?
3. Does the head learn errors representative of the parent's unseen errors?
4. Is a proposed time feature adding information, access to existing information,
   easier optimization, or a shortcut tied to position/composition?
5. Is an online gain due to estimator maturity, adaptation to drift, or another
   component such as the intercept?
6. Does the next architecture need longer state, longer gradient credit, a better
   readout, more updates, or more unique components?
7. Which observations and parameter directions does the actual executed loss
   emphasize after all weighting, normalization, clipping and Adam updates?
8. Does KD add new teacher information, preserve the parent, or both?
9. Does the proposed control actually remove the hypothesized information?
10. Does a candidate beat the untouched competent parent as well as its newly
    trained matched control?
11. What hidden target/mask assumptions are required, and which are independently
    documented rather than guessed from anonymous features?
12. Can the whole experiment reach an interpretable result and, when relevant,
    the served endpoint within the real resource limit?

These questions should narrow the next choice. They are not a requirement to
solve every mechanism before pursuing a strong empirical result.

## Literature transfers and the strongest counterargument

The [source register](../../sources/ASTRA_ULTRA_REVIEW_SOURCES_20261023_R1.md)
records the exact inspected scope: abstracts where only abstracts were read,
sections where full text was inspected, and failed access where no detailed
claim can be made. Most useful transfers are small components:

- GRU-D motivates explicit timing/observation indicators, without equating
  unchanged values with missing data.
- Time-warping, truncated-backpropagation and Transformer-XL work help separate
  row clock, forward access and gradient credit.
- Numerical embeddings and Deep Sets suggest controlled basis/parameter-sharing
  comparisons without proving anonymous-field semantics.
- Beta-NLL separates variance learning from residual weighting; it does not
  establish a gain or remove shared-gradient interactions here.
- Cross-fitting and semiparametric distillation motivate scrutiny of parent
  data reuse, with assumptions different from dependent market sequences.
- Data-constrained scaling motivates separate unique-data and repetition axes,
  without importing language-token ratios.
- Control-engineering lead/lag compensation suggests a bounded response test,
  not a theory of this target or a proof that sharpening helps.

The seeded local-library draw selected Rogii lines 49–64 using seed 2026102310
and an explicitly recorded LCG. Its quadratic-fit residual feature suggests
distinguishing smooth and irregular corrections. The source is a geosteering
writeup, so neither its data semantics nor its reported success transfers.

The strongest objection to this synthesis is that it could overinvest in
diagnosis. A genuinely better model can win before its mechanism is fully
understood. Repeated stress tests, calibration probes and ancestry archaeology
can consume the same budget as a useful new fit, while further exposing the
validation set. A cheap discriminator is valuable only if it changes a decision.

For that reason, the recommendations preserve a bounded exploration path and
the larger teacher direction. They do not impose a universal causal-proof gate,
require public confirmation before locally supported packaging, or treat
uncertainty as failure. A convincing complete-endpoint gain can justify a
follow-up while its explanation remains provisional. Conversely, an elegant
explanation does not justify spending when the proposed test cannot distinguish
it from a simpler alternative.

## Durable record and current disposition

This review is complete as a documentation task. The [ten original reports](README.md),
[40-card register](IDEA_REGISTER.md), [70-question register](QUESTION_REGISTER.md),
[evidence map](EVIDENCE_MAP.md), [corrections](RECONCILIATION.md), source register,
invocation manifest and artifact hashes preserve the work. The hypothesis ledger
records the proposals and their qualifications; the experiment ledger records
a zero-fit review, not a new execution campaign.

The most useful change to research practice is to attach every active idea to
a correction, a credible competing explanation and a decision-changing test.
For the next authorized cycle, recover one missing answer, compare history with
ordinary diversity, and choose one genuinely different exploration question.
All experimental execution remains stopped until explicitly resumed.

