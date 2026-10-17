# State clocks and predictive queries: a narrower literature transfer

Date: 2026-10-17. Status: proposed mechanism refinement, no fitting or inference.
Related cards: H-RR17-03, 04, 08, 18, 25. This note does not add five supposedly
independent architectures to the portfolio. It separates operators that the
same event-clock observation cannot distinguish.

## Four different clocks

| Clock | What changes | Evidence needed here |
|---|---|---|
| Observation | A permitted feature arrives or changes | Observed directly; the 112-feature R1 sample has a change every row |
| State write | Some learned memory is revised | A model intervention, not identifiable from target-change counts |
| Output | The emitted prediction changes | Can be changed on frozen outputs without changing state computation |
| Realized label | The hidden scoring target changes | Retrospective description only; unavailable to the serving policy |

A fifth clock, training credit assignment, governs where gradients are cut. It
does not govern when inference state or outputs update. ARTBP and event-paced
inference therefore address different mechanisms.

Three primary papers illustrate distinct state-write operators:

- [Phased LSTM, Neil et al. (2016)](https://arxiv.org/abs/1610.09513) uses a
  parametrized oscillating time gate that permits cell updates during part of a
  cycle. Its asynchronous-sensor motivation does not establish that row index is
  physical time here, or that the two selected price codes define that gate.
- [Skip RNN, Campos et al. (2018)](https://arxiv.org/abs/1708.06834) learns to skip
  recurrent state updates and permits an update-budget penalty. This changes
  the state trajectory, unlike holding already-computed outputs in our audit.
- [Hierarchical Multiscale RNN, Chung et al. (2017)](https://arxiv.org/abs/1609.01704)
  learns latent boundaries and updates on multiple time scales without supplied
  boundary labels. Its language/handwriting evidence does not establish a
  financial latent hierarchy or a faster competition endpoint.

The checked scope for these three records is their primary abstracts, accessed
2026-10-17. No implementation was downloaded or benchmark reconstructed. The
transfer inference is ours: maintain a small dense observer alongside a slower
memory if the latter is gated. That preserves incoming evidence while testing
whether writing every observation into all memory is wasteful. A negative
output-hold result would not refute this factorization, and a positive result
would not validate state skipping.

## A more specific alternative to copying an arbitrary hidden vector

A teacher's hidden representation is not unique. For example, permuting its
hidden coordinates and transforming the corresponding weights leaves its
input/output function unchanged. A learned projection can address this, so it
is not a correctness bug in hidden-state distillation. But copying all hidden
coordinates can allocate student capacity to information irrelevant to the
two scored targets. Copying only two outputs has the opposite limitation: it
may omit useful information about how the state should evolve.

One coherent middle ground is a **small predictive query state**. Define the
coordinates by a fixed set of observable future questions, rather than teacher
neuron IDs: distributions of the next input event family, event counts over
several fixed row horizons, or changes in selected input groups at fixed event
horizons. At serving time those coordinates are predictions made from the
current causal prefix. Their realized future answers are training supervision
only. They are never fetched from the future by the submitted predictor.

This is related to [Predictive Representations of State, Littman, Sutton and
Singh (2001)](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf),
whose representation uses predictions of future observation tests as state.
Its primary abstract and opening formulation were inspected. The paper's
action-conditional linear representation result does not prove that our
continuous, action-free stream has a low-dimensional sufficient query set.
Our proposal is an engineering approximation, not an application of a proved
sufficiency theorem to this dataset.

The useful change to cards 04/08/18 is precise:

1. Choose a small training-only query bank in advance, spanning row time and
   input-event time. Preserve censoring at sequence ends; a missing future event
   is not a zero-valued answer.
2. Train a causal teacher to answer those queries and the actual targets. This
   remains a future experiment; no teacher is trained in R2.
3. Give the student output KD plus those named query predictions. Compare with
   the same student, teacher, capacity and compute using output KD alone and a
   matched-dimensional projected hidden hint.
4. Judge the final student by the unchanged official WP. Better event prediction
   or a more interpretable state is insufficient.

**Why this may help:** semantics of the intermediate supervision are stable
across teachers, and the query bank may preserve state information that a
two-number output omits. **Why it may fail:** observable future inputs can be
dominated by activity or scale that contributes nothing to the weighted target;
there may be no compact sufficient bank; sparse query losses can compete with
the main loss. Existing event-posterior and temporal-bootstrap negatives are
direct counterevidence against treating this as automatically useful. A renamed
single next-event hazard auxiliary would be a repeat, not this distinction.

**Cheapest next no-fit discriminator:** inspect the exact old auxiliary query
set and its contribution path, then describe the prevalence, censoring and
cross-sequence stability of a fixed candidate bank. Such EDA cannot establish
forecastability or student benefit. Do not fit a probe under the no-training
instruction. Drop queries that merely reproduce the existing next-event
posterior; keep a future fitted comparison conditional on a meaningful gap.

## Why a recent general forecasting model is not a shortcut

[Chronos-2, Ansari et al. (2025)](https://arxiv.org/abs/2510.15821) describes group
attention for sharing information across related variates and covariates,
trained with synthetic multivariate structures. Its primary abstract was
checked on 2026-10-17. This motivates a possible cross-group teacher operator;
it does not supply a compliant pretrained solution to our anonymous-feature,
sequence-reset, no-observed-target-history task. No weights were downloaded or
used. Pretraining eligibility, task alignment and student transfer would all
need separate evidence. The existing cross-stream and local-attention cards
already cover the useful architecture question; creating a new card named
after the paper would add little.

## Decision

Retain the event clock as an input summary unless actual evidence says
otherwise. Keep dense observation, memory writes, output emission and training
credit separate in the experimental comparison. Refine the existing KD/auxiliary
cards toward a semantically specified predictive state only if the old-code
audit shows an actual missing mechanism. The completed R2 output audit rejects
all12 tested causal emission transforms on each of three frozen arrays; it
cannot choose a recurrent architecture.

## Old-code audit correction

The subsequent [prior attack](../reflection/ROUND2_PRIOR_ATTACK.md) established
that a dense observer plus slowly updated memory is already the project's
DualClock design, with mixed historical results. Belief v2 already combined
dense recurrence, event posterior, quiet-row survival and age. The generic
factorization above is therefore not a new proposal. A descendant needs a
specific changed operator and a matched old-design control.

Likewise, next-event queries overlap Belief auxiliaries, `aux_quote`, and older
future-moment supervision. A multi-horizon predictive query bank is retained
only as a conditional refinement after exact query/contribution-path comparison,
not as evidence that semantic supervision was absent. The [final dispositions](../reflection/ROUND3_EDA_DECISIONS.md)
supersede any earlier provisional novelty or priority in this note.
