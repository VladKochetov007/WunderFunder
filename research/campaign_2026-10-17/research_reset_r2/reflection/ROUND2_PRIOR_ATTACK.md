# Round 2: novelty audit against the nearest executed mechanisms

Date: 2026-10-17. This audit compares the three high-risk variants in
[`DEEP_REFLECTION.md`](DEEP_REFLECTION.md) with the executed belief filter,
target-defined hazard, temporal teacher paths, and Wave38 bootstrap. It reads
source and historical records only; it does not run code, fit models, or inspect
new data. The R1 proposal language remains in place as history. This audit
narrows which pieces deserve to remain live.

## Decision first

| Reflection variant | Closest prior | What is truly different | Current status |
|---|---|---|---|
| Duration-bearing latent state | Belief v2's eight-state input-event posterior, held event rates, age readout | A learned, non-geometric hazard for transitions of a latent input regime as a function of that regime's own age and current allowed inputs, including rows with no observed price-code mark. If it is just a new hazard on the same input marks or a target-change hazard, it is a relabeling of prior work. | **Mostly overlap; conditional only.** Reopen only if fixed-model residuals retain a stable state-age pattern beyond current features and the Belief-v2-style posterior. |
| Dense updates plus prediction-innovation readout | Every existing GRU already updates densely; output hold/cadence/EMA are fixed-prediction interventions; temporal teacher/reanchor objectives already exist | An explicit two-value prediction accumulator with a learned, input-responsive innovation is a parameterization choice. It does not add observations or inherently add information beyond a sufficient recurrent state. | **Do not sell as a missing consistency objective.** Consider only if fixed-output diagnostics show a reproducible emission trade-off that a causal, input-responsive readout could plausibly resolve. |
| Rao–Blackwellized filter | Belief v2 already filters a finite discrete posterior; prior target-label denoising is a different approximation | Exact conditional marginalization of a specified continuous substate, such as a linear-Gaussian state conditional on a discrete regime. Belief v2 does not carry or marginalize that continuous posterior. | **Mathematically distinct but unsupported.** Keep closed until an explicit tractable factorization and residual evidence for its latent mixture both exist. |
| Semantic future-observable query hints | Belief v2's next-mark/waiting-time posterior; current event/quote auxiliary heads; the old future-moment probe; cards 04/08/18 | A fixed, named bank of multi-horizon future input questions used as intermediate KD targets, after removing targets that existing heads or the event posterior already represent. | **Narrow refinement, low prior.** It is not a new architecture or a generic future-input SSL win; only the nonduplicated bank and a matched output-KD / hidden-hint control remain untested. |

The three variants are not three independent novelty claims. The first is
substantially covered by prior input-event filtering; the second is an emission
parameterization adjacent to prior output and teacher smoothing; only the third
is a different inference operator, but its necessary model structure has not
been established. The semantic-query refinement is likewise a narrower target
choice within existing hint/auxiliary families. No variant is promoted by the
1,253 target-clock exceptions.

## What Belief v2 already did

The archived [architecture memo](</Users/nb/Documents/wunder/research-2026-10-07/execution/next_belief/ARCHITECTURE_MEMO.md>) and
[filter implementation](</Users/nb/Documents/wunder/research-2026-10-07/execution/next_belief/belief_core.py>) describe an eight-state causal posterior over **observed input-event marks**. The marks use selected raw code-change flags for the two instruments. A bounded event rate for each latent state is projected from the GRU output at the preceding marked event and held until the next marked event. On a quiet row, treatment multiplies the posterior by the per-state no-event likelihood; on a marked row, it uses event-rate and mark likelihoods followed by a learned state-transition matrix. The `noquiet` control leaves the posterior unchanged on quiet rows. Both arms still run the ordinary GRU through every row.

The filter carries the posterior, held rates, and rows since the latest marked
event. A zero-initialized readout of posterior and log age adjusts the ordinary
two-target head; a fixed-weight auxiliary teaches the posterior to predict the
next observed input mark. Thus it already combined a probabilistic event
posterior, quiet-survival evidence, elapsed-row age, and a separate forecast
readout. It was not a hard freeze of the whole hidden state or an output-hold
rule.

The duration distinction is narrow but real. Belief v2's hazard is the chance
of the **next observed input mark** conditional on a latent state and a rate
held from the preceding mark. It has a memoryless per-row hazard between marks;
the scalar age is also supplied to the output readout. The proposed
semi-Markov state would instead need a duration law for **latent regime
transitions**, for example

\[
P(z_{t+1}\ne z_t\mid z_t,a_t,X_{\le t})=h_{z_t}(a_t,X_t),
\]

where (a_t) is time since the inferred regime began, and a regime may persist
or transition even when the selected code marks do not fire. That is not a
new name for the existing mark survival posterior. It is also not warranted by
the exception census: an imperfect target clock says neither that a latent
regime changed nor that this change was forecastable.

The main implementation is especially important for interpreting the
`noquiet` contrast: the quiet path changes only the added posterior, not whether
the GRU consumes the row. Therefore the executed result does not test “skip
hidden updates on quiet rows.” Belief v2 treatment versus fresh ordinary was
−0.000003 strict-valid WP (95% chain interval [−0.000027, +0.000019]) and
+0.000020 on the train diagnostic ([−0.000026, +0.000067]); `noquiet` was
−0.000004 and +0.000005, respectively, with both intervals crossing zero.
The unchanged parent was above all three continuations. This closes that fixed
eight-state, mark-conditioned, one-epoch recipe as a demonstrated improvement;
it does not refute all duration models. It does make a same-clock posterior
re-run low priority. The exact pooled/target result receipt is
[belief-seed3-eval.json](</Users/nb/Documents/wunder/research-2026-10-07/execution/results/belief-seed3-eval.json>).

The older [stage3 hazard source](</Users/nb/Documents/wunder/research-2026-10-07/execution/idea3/survival_aux.py>) is different in a decisive way: its event is defined from the **future target sequence** (pair change or target contrast change). It constructs right-censored training labels and applies a training-only survival loss. It does not provide a serving-time event detector or a carried posterior over input states. Its predictor cannot use those labels at inference. The recorded hazard-versus-original forecast delta was −0.000082 strict and −0.000031 train diagnostic. Do not present a future target-change hazard head as an untried input-state duration mechanism.

## Temporal consistency and future-run smoothing are not missing ingredients

The current [main loss](../../../../src/wnn/training/losses.py) already has
several distinct temporal teacher paths: `distill_boot_h` can substitute a
future teacher prediction while the raw target pair stays unchanged through
that offset; `distill_alpha_far` changes teacher mixing by distance to a target
change; `distill_reanchor` calls a separate blended-target rule; and `td_alpha`
can use the EMA teacher at a later target-change location. These paths are not
all active together in every recipe, and they alter optimization targets rather
than serving-time emissions. Their existence nonetheless rules out the claim
that future-run consistency is simply absent from training.

For fixed clipped labels and fixed predictions, weighted squared error already
separates a constant-label run's mean error from its within-run prediction
variation:

\[
\sum_{t\in r} w_t(p_t-y_r)^2
=W_r(\bar p_r-y_r)^2+\sum_{t\in r}w_t(p_t-\bar p_r)^2.
\]

This identity is recorded with its mask and clipping limits in the parent
[persistence note](../theory/PERSISTENCE_AND_PREDICTION.md). A generic penalty
for prediction fluctuation inside fixed-label runs is therefore not a newly
discovered loss term; it reweights parts of the existing squared-error
objective and can suppress useful updates. The identity does not make WP
additive, imply that every distillation target equals the fixed label, or show
that a causal model can reproduce a future-informed run average.

Wave38 is the closest historical test of the future-run teacher idea. With the
Wave19 recipe held fixed, it replaced the hard target with the teacher's
prediction at (t+8) only when the raw target pair remained unchanged through
that point; the current teacher still contributed 0.7 and the guarded future
teacher 0.3. It used two seeds and the reused strict-valid425 panel. The
bootstrap arm beat the weaker teacher-only arm by +0.001476, but trailed the
stronger family arm by −0.000758 (conditional 95% interval [−0.001757,
+0.000200]); both seed-specific bootstrap-versus-family point estimates were
negative. This is material counterevidence to generic future-run smoothing at
that stage, not a universal rejection of teachers or all training objectives.
The [source-bounded historical row](</Users/nb/Documents/wunder/audit-2026-10-17/historical-wave-results.json>) and [local history](</Users/nb/Documents/wunder/audit-2026-10-17/LOCAL_HISTORY.md>) preserve the comparison's exposure and lineage caveats.

The prediction-innovation readout remains distinct from Wave38 in *where* it
acts: it changes the causal emission map and can take new current-row inputs;
Wave38 changed a training target. It remains close in *motivation* to output
smoothing and temporal teacher methods. Since the prior prediction is itself a
function of the available past and model state, an explicit (P_{t-1}) input is
an inductive bias or compact accumulator, not new information. It should not be
pitched as adding a persistence penalty that the current objective lacks.

## Questions the pending EDA can actually divide

| New result | If observed | Priority change | What it cannot establish |
|---|---|---|---|
| Per-target label reconstruction from the approximate input clock, including persistent error duration and clipped target-weight mass | Little cumulative weighted distortion, and no long residual tail after clock misses | Keep the clock as a useful approximate summary; close exception-driven duration variants for now | Exact sufficiency of the clock, or a stochastic ceiling |
| Same reconstruction | Persistent, nontrivial weighted distortion beyond the individual exception rows | Raise the question of a richer causal state, but only if a fixed predictor's errors align with those periods and contemporaneous allowed inputs provide a signal | That target persistence is a serving-time state, or that hard/soft event gating helps |
| Full-prediction-panel AND/OR hold, cadence and EMA controls | Repeatable full-panel WP gain across incumbents, without concentrating damage on scored exceptions | Raise a simple emission policy; do not infer hidden-state freezing or a new latent regime | A trained innovation head's gain, model-state savings, or public transfer |
| The same transforms | Holding/EMA lose while label-plateau oracle means reduce within-run variance | Lower priority of generic emission smoothing; investigate whether current input/readout learns useful progressive estimates | A realizable causal oracle gain, Bayes ceiling, or proof of forecastable exceptions |
| The same transforms | Oracle gain is negligible | Lower priority of within-run output fluctuation as the main deficit for these fixed models | That model state, training, or information is optimal |
| Exceptions joined to already-existing fixed predictions and allowed-feature change summaries | Errors recur on high-weight misses, with other current inputs moving in a reproducible way | A soft, feature-conditioned state update becomes more plausible than target-clock repair; compare against the existing mark posterior | A new label-derived serving gate or physical causal effect |
| Any residual summary | Residuals are multimodal after conditioning on current features, sequence position and duration, with repeated regime-specific structure | Only then define a candidate discrete/continuous state partition for Rao–Blackwellization | That a linear-Gaussian conditional model exists or is correctly specified |

Counts alone will not change these priorities. If the reported EDA has only
aggregate weight shares and output transforms, it can separate clock rarity
from emission filtering but cannot identify a latent regime or a
Rao–Blackwellizable substate. Treat the global WP score, exact mask, clipped
weights and prediction lineage as the unit of evidence; do not infer a score
effect from an isolated row count or a retrospective class.

## Predictive-query refinement: narrower novelty, substantial prior overlap

The sibling [state-clock and predictive-query note](../theory/STATE_CLOCKS_AND_PREDICTIVE_QUERIES.md)
refines existing cards 04, 08 and 18; it is not an extra architecture family.
Its proposal is to supervise a causal student on a predeclared set of semantic
questions about future **observable inputs** (for example, event-family
probabilities or feature-group changes over several horizons), alongside the
same target/output KD. The future values are training targets only. That is
meaningfully better specified than “copy these arbitrary teacher neurons,” but
it remains an auxiliary representation-transfer objective. It adds no
inference-time observation and cannot add information absent from the causal
prefix.

The exact overlap audit is:

- **Belief v2:** the input-event posterior already predicts a next observed
  mark, models its waiting time through per-state rates and quiet survival, and
  exposes posterior and event age to the forecast readout. A query for the same
  next-mark family or its one-step survival probability is not new. Longer
  multi-event counts or feature-group outcomes can be distinct, but only if
  their conditional targets are not recoverable from this existing posterior
  and are not just a different binning of its held-rate process.
- **Existing next-change heads:** `aux_next` and `aux_next_k` in
  [`targets.py`](../../../../src/wnn/training/targets.py) predict future
  *target-contrast* change time/sign, not future allowed-input queries. They
  are not an exact duplicate, but any query bank defined from target changes
  would collapse back onto that older target-defined family and remains
  training-only.
- **Existing future quote auxiliary:** `aux_quote` has a declared
  class-3-only target of mean quote displacement after 8/24 book events
  ([head construction](../../../../src/wnn/training/setup_model.py),
  [loss path](../../../../src/wnn/training/losses.py)). A query bank containing
  those same price-displacement questions is a duplicate unless it changes
  the target definition and proves that the existing objective was absent from
  the exact parent lineage. The code path's existence does not imply the
  current model enabled it or establish a score result.
- **Old privileged-moment probe:**
  [`idea4/moment_probe.py`](</Users/nb/Documents/wunder/research-2026-10-07/execution/idea4/moment_probe.py>)
  formed +8-row future volume/activity differences for teacher-only
  predictions of target-weighted numerator/denominator moments. Its students
  used frozen incumbent outputs plus current raw features, random-feature
  ridge heads, and a fixed blend; it was neither a recurrent query-state
  teacher nor hidden-state KD. It lost to the incumbent by 0.001001 valid and
  0.001116 train, while narrowly beating its weaker causal teacher control.
  This is not the same query bank, but it is negative task evidence against
  assuming that privileged future observables transfer usefully through an
  auxiliary.
- **Cards 04/08/18:** card 04 is projected hidden-state matching, 08 is a
  next-input-event family/waiting-time auxiliary, and 18 is contrastive future
  latent prediction. A named query bank changes the target semantics, not the
  core distinction between output KD, hidden hints and future-input
  supervision. In particular, if the bank is only next event type plus wait,
  it duplicates both card 08 and Belief v2; if it is merely a contrastive
  latent from future inputs, it returns to card 18.

So the novelty is limited to the **specific semantic query bank and its
matched transfer test**, not “predicting future inputs” in general and not a
new state architecture. A useful matched comparison would hold teacher,
student, target/output KD, capacity and compute constant across output-KD-only,
same-dimensional projected hidden hint, and semantic-query hint. The bank
must first remove targets already covered by Belief v2, `aux_quote`, or other
active heads. No such comparison is authorized or run here.

The pending EDA can only screen this idea at the design level. It can show
whether candidate event/feature queries are nondegenerate, how often they are
censored, and whether their definitions collapse to the old mark or quote
targets. It cannot determine conditional mutual information with the target,
whether the student can transfer the information, or whether WP improves; those
would require a separately authorized fit/probe. Do not elevate the query bank
solely because a future query has nonzero variance or because an observable
changes on most rows.

## Conditional disposition

1. **Duration state:** keep only the explicitly different latent-regime-age
   question. First require fixed-model residual evidence beyond current inputs
   and the previous-event/age posterior. If evidence is merely that target
   labels sometimes change during a selected-code quiet interval, stop; that
   repeats the clock exception under new terminology.
2. **Prediction innovation:** do not carry forward generic smoothing or
   consistency as a novelty claim. Await the predeclared frozen-output
   interventions. A trainable input-responsive accumulator would need its own
   causal readout hypothesis and controls against the existing recurrent head,
   fixed EMA/hold transforms and temporal teacher paths.
3. **Rao–Blackwellization:** close for this campaign unless a concrete
   conditionally tractable latent substate is specified from the allowed-input
   process and supported by stable residual structure. A discrete posterior,
   multimodal target histogram, or prior label-denoising failure alone is not
   such a factorization.

No new fit follows from this audit. Keep the pending EDA descriptive and use
its result to answer the table, not to invent a post-hoc fourth interpretation.
