# Round 1: bottlenecks before proposal review

Date: 2026-10-17. This framework was written before reading the 25 proposal cards or their R2–R4 reviews. It uses the official data and scoring contract, the current model catalog, and the R1 EDA summaries. It makes no fit, score, or causal claim.

## Task as an information and estimation problem

For each sequence, let \(X_t\in\mathbb R^{112}\) be the currently observed row, \(H_t=(X_0,\ldots,X_t)\) the allowed causal history, \(Y_t=(t0_t,t1_t)\) the hidden-at-inference target, \(S_t\) the model's internal state, and \(P_t\) the emitted prediction. The competition requires an output at every eligible row, including rows that will not be scored. It resets state at sequence boundaries and forbids future rows, target labels, scoring masks, sequence-id information, or cross-sequence state at inference.

The deployed computation can be factored as

\[
S_t = U(S_{t-1}, X_t),\qquad P_t = G(S_t,X_t),
\]

with sequence reset, followed by the fixed scoring transform

\[
\tilde Y_t=\operatorname{clip}(Y_t,-2,2),\quad
\tilde P_t=\operatorname{clip}(P_t,-2,2),\quad
w_t=|\tilde Y_t|.
\]

The metric pools weighted centered moments over all selected rows for each target, then averages the two correlations. Therefore a rare event's count alone does not bound its effect: its clipped target magnitude, prediction error, co-movement with other errors, and effect on global centering and variance all matter. A class's share of total target weight is not an additive share of score.

Four broad limits should be kept separate:

1. **Information / observability:** even the best causal predictor of \(Y_t\) from \(H_t\) may be limited by hidden future shocks or insufficiently informative observations. This is a limit of the permitted observation stream, not necessarily a model defect.
2. **State / representation / compression:** a finite or lossy encoding \(S_t\) may discard predictive information present in \(H_t\). Dense row updates, event-driven state updates, approximate recurrence, and exact repeated-row skipping are different encodings.
3. **Estimation / optimization:** finite data, architecture, losses, teacher targets, initialization, and optimizer trajectory may keep a fitted predictor below the best predictor representable by the chosen model. This is distinct from missing information.
4. **Emission / measurement:** a correct state can still be mapped poorly to outputs, clipped in a score-relevant way, or assessed with a diagnostic that confounds global WP contributions. Output timing and hidden-state updating are separate decisions.

The evidence currently measures data descriptions and a handful of already-existing model/catalog outcomes. It does not estimate a Bayes ceiling, identify a sufficient state, or measure the realized residuals of the 25 ideas.

## Keep the mechanisms distinct

| Mechanism | What it means here | What the 1,253 exceptions do and do not establish |
|---|---|---|
| **Label persistence** | Temporal persistence or switching of the target sequence itself. Labels are training/evaluation observations; they are never available as lagged inputs at deployment. A label persistence statistic can motivate a predictive prior, but feeding true prior labels into the serving state would be leakage. | A target change when a selected input clock does not fire rejects a universal deterministic target-update rule. It does not measure label transition hazard, persistence length, or achievable prediction. |
| **Hidden predictive state** | Predictive memory of permitted inputs, such as a filtered book/trade regime, event age, burst intensity, or cross-instrument context. This state must be inferable from observed history and updated within each sequence. | The clock is only the conjunction of `i0_p0` and `i0_p11`. The full-feature sample says some other input changes on every adjacent pair in 256 sequences. So an exception is not evidence that the full observation was unchanged or that hidden input state was unobservable. |
| **Causal feature updates** | How the model state incorporates every arriving row. An event-triggered update can be a computational/representation choice; a soft decay or duration update can keep state evolving during quiet periods. | Exceptions falsify hard state-freeze or exact target-hold rules keyed only to that conjunction. They do not show that the underlying model skipped updates: the official interface requires updates on every row, and an output may be held while hidden state still updates. |
| **Emission policy** | How the current state maps to the required per-row prediction: emit from every state, hold the previous output, decay it, or use an event-conditioned readout. | The grader requires a prediction every eligible row and hides the scoring mask. Sparse output submission is invalid. A held prediction is permissible only as a prediction policy; it says nothing about whether internal state continues to consume all inputs. |
| **Observability / stochastic ceiling** | Uncertainty in future targets conditional on all allowed observations \(H_t\), including unpredictable events. In principle this sets a ceiling for any causal model and optimizer. | Input-clock exceptions do not estimate \(P(Y_t\mid H_t)\). They may reflect other changing features, persistent latent state, an unobserved shock, or label noise. No one of these is selected by the census. |
| **Actual model residuals** | The error pattern of fixed, causally generated predictions, inspected under the official mask and target weights. Errors can be concentrated by event age, class, target side, sequence, or exception status. | Exception counts alone cannot show that current models miss those rows. Required evidence is a pre-existing prediction cache joined to the exact census and evaluated as diagnostics, not a newly fitted gate or calibrator. Any final claim must re-score the whole panel because WP is globally centered. |
| **Teacher–student optimization** | Whether teacher supervision, target construction, capacity, gradients, or optimization transfers useful causal predictions into the final student. It concerns learning the mapping/state, not adding information unavailable to both teacher and student. | The model catalog shows materially different input and teacher lineages (e.g., 211-input distilled and 263-input tied-lead GRU families), but those descriptions do not isolate a distillation bottleneck or show its relation to exceptions. Distillation cannot recover information absent from inputs unless its training targets act as useful regularization/optimization signals. |

## Re-read the event finding at the right scale

The census reports 941 train and 312 valid need-eligible target-change rows outside the selected `i0_p0 AND i0_p11` conjunction; 92 valid rows are officially scored. Against all eligible target-change rows, these are about 35 and 48 per million respectively. Against all 249.6 million scanned rows, all 1,253 are about 5 per million. The exceptions are therefore sparse. They are nevertheless counterexamples to a universal implication, and the sparse rate does not imply negligible WP impact.

The denominator matters: the scan compares target changes with just two selected price codes, not with all 112 current features or with the model's history. In the separate 256-sequence sample, no adjacent *complete* feature vectors were equal. That observation is sample-limited but directly blocks the inference “selected clock quiet ⇒ no new observation.” The 52 enumerated examples were all need-eligible and tick-sized; that is a first-in-order example set, not a random estimate of all exception amplitudes. One scored example has large simultaneous target changes, but one example cannot establish the distribution. The 92 scored exceptions are row transitions, not independent statistical trials or necessarily distinct episodes.

This yields a narrow correction: preserve an approximate event clock as a candidate summary if useful, but neither call it an exact target clock nor assume that output holding implies hidden-state freeze. A rare exception is a hard counterexample to a universal law and only a weak clue about average predictive value.

## Compression and time: what is actually supported

Literal adjacent-row run-length compression of the complete 112-vector has no observed opportunity in the 256 sampled sequences. That says nothing about sparse component deltas, low-rank updates, recurring states separated in time, or a learned compressed memory; those schemes can preserve changes while reducing recurrent computation or history storage. Conversely, a price-code event clock is a lossy compression of the input stream. Its usefulness depends on retained target-relevant information, not on exact target equality.

The official callback advances causally row by row, so any event-time/semi-Markov mechanism must define a causal duration or hazard state and must still process all rows. A Rao–Blackwellized estimator would be a way to integrate a tractable part of latent-state uncertainty analytically while sampling or learning the rest; it is not automatically a better predictor or a free compression method. Whether either helps depends on a justified state factorization and measurable prediction residuals.

The EDA found that 43.69% of retrospective class-3 target magnitude weight occurs at age ≥64 relative to the selected joint price change, and that class contributes 47.69% of total valid target magnitude weight. This makes long quiet histories worth checking in existing predictions. It does not establish an age effect: the age is not time since target change, the class is hindsight-defined, and global WP prevents treating that weight share as score contribution.

## Evidence that would change the priority order

1. **For event-time or persistent-state mechanisms:** with already-existing causal prediction caches, compare exception and non-exception weighted residuals and complete-panel WP under predeclared event-age bins. A sizeable, repeatable residual concentration on quiet-age / non-clock-change scored rows raises their priority; near-zero residual there, or the same error profile on clock-change rows, lowers it. Do not fit bins or a gate to those results.
2. **For feature coverage versus unobserved shocks:** establish whether exception rows coincide with events in other allowed features using the saved feature sample or an already-authorized descriptive artifact. If the full observable stream changes, prioritize its state encoding or feature coverage; if the full vector is unchanged yet targets move, the observation/label stochastic-ceiling explanation gains weight. Current evidence does not census all 112 inputs on all exceptions.
3. **For long memory / sequence compression:** ask whether residual structure survives beyond the current 99-row warm-up and short event lags, after conditioning descriptively on allowed current features and sequence exposure. Long-age residual concentration favors durable state; no such concentration weakens expensive long-memory designs. Exact row repetition is already a poor rationale.
4. **For cross-instrument event-time state:** the 64-sequence fixed-lag control is heterogeneous across retrospective classes. More sequences with the same pre-fixed lag and phase-shift definitions must preserve direction and association before a universal `i1 leads i0` story moves up. A mixed result leaves asymmetric/conditional effects open, but does not justify label-derived serving regimes.
5. **For teacher–student / optimizer mechanisms:** matched existing teacher and student predictions, ancestry, and seed/fold residual correlations would reveal whether error is shared (information/data limit or common representation) or differs (capacity/optimization/distillation opportunity). A catalog row or one short architecture screen cannot resolve that.

## Starting causal sketch

```text
past/current allowed inputs H_t ──► latent predictive state Z_t ──► Y_t
            │                             │                         │
            ├──► encoded state S_t ───────► prediction P_t ─────────┘
            │              ▲                    ▲
            │              │                    │
            └──► update/forget rule             readout/emission rule
                           ▲                    ▲
                 model capacity & training / teacher targets

hidden shock U_t ───────────────────────────────────────────────► Y_t
```

This is a hypothesis map, not a discovered causal graph. In particular, training on \(Y\) estimates associations under the observed train distribution; it does not establish that a book event causes the target or that the target's anonymized horizon is one row.

## Source basis and limits

- Official interface/data contract: [`docs/data_overview.md`](../../../../docs/data_overview.md), [`docs/submission_guide.md`](../../../../docs/submission_guide.md).
- Official scorer: [`src/wnn/gmetric.py`](../../../../src/wnn/gmetric.py), metric ID `global_wp_clip2_v1`.
- Current model lineage and endpoint distinctions: [`research/unified/models/CURRENT_MODEL_CATALOG.md`](../../../../research/unified/models/CURRENT_MODEL_CATALOG.md).
- Direct EDA basis: [`research_reset_r1/eda/FINDINGS.md`](../../research_reset_r1/eda/FINDINGS.md), [`EVENT_LAG_CONTROLS.md`](../../research_reset_r1/eda/EVENT_LAG_CONTROLS.md), and [`research_reset_r1/REPORT.md`](../../research_reset_r1/REPORT.md).
- Scope and evidence limits: [`research_reset_r1/CONTRACT.json`](../../research_reset_r1/CONTRACT.json), [`specs/current-state.md`](../../../../specs/current-state.md), and [`conduct-creative-research/SKILL.md`](../../../../.agents/skills/conduct-creative-research/SKILL.md).

## Addendum after review: scope correction and causal graph

This addendum preserves the Round 1 wording above and narrows two overstatements identified during review.

The 1,253 counterexamples falsify only the exact implication that the selected `i0_p0 AND i0_p11` change event is necessary for every eligible target transition. They do not establish that predictive information was lost, that any model used a hidden-state freeze, or that a hard feature-triggered update rule would lower official WP. Likewise, observing some other full-vector feature change on every sampled adjacent pair only shows that the two-column clock is not a complete description of sampled input updates; it does not prove those changes carry useful target-predictive information. Approximate event summaries remain an empirical question.

The diagram above should be read with the following corrected causal structure. Model predictions do not cause fixed labels; the official scorer receives both, with the hidden selected mask and clipped target-derived weights. During fitting, permitted training labels and inputs jointly affect learned parameters.

```text
allowed input history H_t ──► latent predictive state Z_t ──► fixed target Y_t
            │                                             │
            ▼                                             ▼
causal model state S_t ──► prediction P_t ──┐       training loss / fitted θ
                                            ├──► official score(Y, P, M)
official mask M and target weights ─────────┘

hidden shock U_t ─────────────────────────────► Y_t
```

This is a bookkeeping graph, not a discovered data-generating graph. `Z_t` and `U_t` are explanatory placeholders; no EDA here identifies them or estimates a stochastic ceiling.
