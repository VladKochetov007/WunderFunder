# Learning critique round 2: adversarial review of microstructure proposals

**Scope.** This is a no-fit review of the twelve cards in [MICROSTRUCTURE_ROUND1.md](../sources/MICROSTRUCTURE_ROUND1.md), the existing [EDA summary](../eda/summary.json), and its [audit implementation](../eda/stream_audit.py). I ran no training, fitting, inference, submission, or additional EDA. The three synthesis directions below are hypotheses and triage choices, not estimated gains.

## Evidence boundaries that change the ranking

The EDA sampled 32 whole sequences per retrospective label stratum in each split (256 sequences, 5.12 million rows total), with equal-stratum sequence selection; this is not the natural pooled mixture and does not create a new holdout. Every stratum was exposed. Strata are label-derived diagnostics only: they are never available inputs, serving gates, or sequence identities. Marginal train/valid drift within a stratum is descriptive and can be confounded by which sequences were sampled. The largest reported i1_volume marginal JS values are 0.2349 for stratum 1 and 0.1773 for stratum 2, but 32 sequences per stratum cannot distinguish a stable drift from sequence heterogeneity. This is a caution on portability, not evidence for an i1 predictor.

The audit’s event definitions are exact component-change proxies, not verified exchange events. Extras changed on 639,968 of 640,000 rows in each sampled stratum, so an “any feature changed” trigger is effectively always on. The separate root audit also reports **zero identical full 112-feature rows**: whole-row run compression is ruled out. That does not rule out useful lower-dimensional state compression, but it kills the proposed premise that repeated full rows supply a cheap event clock.

A critical update to the earlier uncertainty in the source note: the root follow-up confirms 52 scored-eligible target-change exceptions outside the i0_price_and conjunction, all at least one tick in size; many have both selected i0 price components unchanged. One cited valid example is row group 1384, step 14,166, with target changes [-1.6893,+2.17197]. The exact-condition law therefore fails, even though these exceptions are rare (about 0.0055% of the audited changes). Do not dismiss them as float jitter, freeze the target on those rows, or infer a universal target trigger. This keeps the local target-reconstruction result **partial**. Nor does the exception identify which anonymous input component caused the change.

Clipped-target correlations across the four retrospective strata (about -0.65, -0.75, -0.89, and -0.90) are real descriptive dependence, but the stratum trend is not a deployable class rule and does not prove a fixed target equation. The existing age screen found finer hold-age resolution without predictive evidence; valid scored weight mass at age ≥64 was 0.0616, 0.095, 0.2288, and 0.3627 by the same retrospective strata. These are outcome-weight distributions, not evidence that an age feature improves the official pooled weighted-Pearson score. The prior field-age result and partial-law failure remain in force.

## Card-by-card disposition

### 1. Self- and mutual-excitation traces — **Revise; diagnostic only**

**Novelty / equivalence.** A fixed exponential Hawkes trace is an EMA of event indicators; a small bank is a bank of causal filters. A low-rank excitation matrix is a bilinear cross-filter and overlaps event-age, flow-surprise, and small SSM/TCN features. The potentially distinct element is a learned cross-family excitation interaction, not the existence of traces.

**Semantics / causality.** Exact changes in broad feature groups are not validated event types or signed order flow; the extras rate shows why raw change flags need care. A trace updated from x_t and prior rows is causal and resettable. A “next-row event rate” is an offline outcome diagnostic only.

**CPU route / support / falsifier.** A few scalar recurrences are CPU-cheap; teacher distillation is unnecessary until a larger model demonstrates value, and would not rescue a non-informative event vocabulary. No measured clustering support is reported. First require same-family dependence to beat a within-sequence shift control and persist in both halves, excluding the near-always-changing catch-all channels. If this descriptive gate fails, stop.

### 2. Duration-surprise clock — **Defer; reopened age family**

**Novelty / equivalence.** Past-normalized elapsed time is not identical to raw age, but it is a rate-adaptive reparameterization of the same clock; with an event trace it may collapse to another intensity feature. It substantially overlaps the old shared/field-age clocks.

**Semantics / causality.** Row intervals are not wall-clock durations, and the event family is still an unverified feature-change proxy. The ratio is causal only if its expected duration uses completed past intervals and is reset at sequence boundaries.

**CPU route / support / falsifier.** It needs only a few counters and summaries. There is no measured predictive support; the prior screen established representation resolution, not utility. Cheapest no-fit gate: on forward feature segments, compare event-family interval/rate separation from raw elapsed rows versus the proposed past-only normalized age, with sequence-local baselines and no label selection. If scale estimates are unstable or do not improve that feature-only separation, do not reopen age modeling.

### 3. Power-law memory over side proxies — **Defer**

**Novelty / equivalence.** A logarithmic bank of exponentials approximates a power-law kernel, so this is primarily a choice of decay spectrum and can be represented by a multi-pole SSM. It is not independent of the multi-trace proposals.

**Semantics / causality.** “Bid side,” “ask side,” and signed order activity are unsupported interpretations of anonymous volume-like channels. An unsigned innovation norm cannot recover order sign. A proxy built from past/current features is causal, but calling it order-sign persistence would exceed the evidence.

**CPU route / support / falsifier.** A small recurrence bank can run cheaply; compressing an unsupported proxy from a teacher has no rationale. The source note already records weak flow screens and an in-sample-positive / chain-half-negative volume-code probe (+0.0062 / -0.0078). Cheapest no-fit falsifier: test the feature-only proxy’s lag dependence across sequence halves and against within-sequence shuffles, without a side interpretation. An unstable slope or no persistence beyond short lags ends the branch.

### 4. Directed i1-to-i0 lag surface — **Defer; explicit prior family**

**Novelty / equivalence.** The note correctly calls this overlap with prior i1-flow/cross-impact work. A fixed lag surface is a sparse cross-correlation kernel; its low-rank form is a bilinear temporal interaction or small causal convolution. Its only plausible delta is resolving the lag profile beyond cumulative flow since an i0 event.

**Semantics / causality.** There are no supplied exchange timestamps, so lags are row-step approximations, not synchronized event time. Use stable feature IDs and no unverified direction/aggressor semantics. The serving feature must consume only i1 through t to predict at t; i1 at a later row is legal only as an offline feature-event outcome in the no-fit diagnostic.

**CPU route / support / falsifier.** A handful of lag recurrences is cheap and a larger causal teacher could be distilled to them, but that is premature. Static i1 cross-impact has no reproducible local gain, and the volume drift flags portability risk. Require predeclared directed cross-correlograms to beat sequence-local circular shifts and remain stable across within-sequence halves. A surviving feature lag is still not forecast evidence.

### 5. Low-order path signatures — **Revise into a small lag-interaction diagnostic**

**Novelty / equivalence.** The antisymmetric second level is a real order-sensitive statistic, but for a low-dimensional path it is a structured sum of pairwise lead-lag interactions. It is therefore close to card 4’s lag kernel and existing bilinear proposals, not a wholly separate family. Without an explicit time coordinate, signature invariance can also discard row/event timing that matters here.

**Semantics / causality.** Raw anonymized channels can be retained without claiming queue meaning, but the grouped marks and normalization still need justification. A prefix/window signature is causal if it uses only rows through t, resets per sequence, and never centers on a future event.

**CPU route / support / falsifier.** Degree two in a tiny stream is incremental and inexpensive; it is directly implementable without a teacher. There is no measured order-area support. The proposed shuffle preserving counts/sums is a good no-fit falsifier, but add the key equivalence check: does the area add stable information over a small fixed cross-lag basis? If not, keep the simpler basis.

### 6. Continuous covariance state — **Continue only as a redundancy diagnostic**

**Novelty / equivalence.** A small EW covariance is a bank of pairwise EWMAs. Trace, eigenvalue, and log-determinant are summaries of that same second-order state; they are not separate learned regimes. This is a useful alternative to discrete bins, but its novelty is limited.

**Semantics / causality.** Group-level innovations must be formed without inventing physical feature meaning. Past-only updates and sequence resets are straightforward. Retrospective strata must not become a covariance-state selector.

**CPU route / support / falsifier.** A 4-by-4 recurrence is cheap, with no necessary teacher. There is no evidence in the summary that covariance orientation adds beyond total activity. Cheapest no-fit falsifier: compare its continuous variation and redundancy against a scalar causal magnitude EWMA, separately by sequence; if it collapses to activity or its direction is unstable, defer it. Older negative/null regime tables remain a reason not to promote it to a regime model.

### 7. CUSUM/FOCuS gate — **Continue only as a feature-only stress test**

**Novelty / equivalence.** The process-monitoring transfer is genuinely cross-domain, but fixed CUSUM is reflected accumulation of standardized residuals and can reduce to a thresholded EWMA or event count. FOCuS’s richer scan is not the same cheap scalar state.

**Semantics / causality.** A shock alarm on anonymous, past-standardized innovations is semantically modest. Past-only scale, sequence reset, and no prediction suppression are necessary. A detector must not be tuned by retrospective class or validation target.

**CPU route / support / falsifier.** Fixed CUSUM is CPU-cheap; the more complex scan needs a measured benefit before inclusion, and a teacher is not required. No real-shift evidence is in the EDA. The note’s injected-shift stress test is an appropriate no-fit falsifier: set a threshold procedure in advance, then measure false alarms on untouched feature segments and detection delay on synthetic copies. Failure on ordinary clusters or on the injected shift defers the model, not just its threshold.

### 8. Anonymous-component interaction graph — **Defer**

**Novelty / equivalence.** HLOB/graph and bilinear families are already in the local catalog; the source note itself says this is not novel. With a fixed graph, one message-passing layer is sparse linear mixing; feature-pair gates add bilinear terms. A stable correlation graph is not by itself a predictive graph.

**Semantics / causality.** The warning against index adjacency is essential. No graph edge should be named as depth, queue order, or economic influence without a verified mapping. Training-sequence-only graph construction is compatible with deployment; cross-sequence hidden state is not.

**CPU route / support / falsifier.** The proposed model/artifact cost is medium-high, substantially above a sparse interaction student. There is no confirmed HLOB result or measured graph stability. Cheapest no-fit falsifier is the proposed two-half edge-stability test against a within-sequence shuffle, with no validation-target construction. If edges are unstable, do not pay for a graph teacher or student.

### 9. State-first, flow-second gate — **Defer; gated bilinear re-expression**

**Novelty / equivalence.** State-conditioned flow is a meaningful question, but a multiplicative gate is a low-rank bilinear interaction and overlaps the older queue/refill/flow-surprise families. It must show conditional event structure beyond generic activity to be distinct.

**Semantics / causality.** This avoids named queue states if it uses raw stable columns, but any feature bins must remain descriptive rather than physical labels. Use current and past features only; never condition on a later i0 transition.

**CPU route / support / falsifier.** A low-rank gate is feasible on CPU and could be distilled, but static-flow negatives make the teacher unjustified now. No target or feature evidence for a state-conditional event rate is shown. Cheapest no-fit falsifier: conditional event-rate tables with fixed robust current-state summaries and activity-matched bins. If the apparent gate is only an activity-scale effect, stop.

### 10. Next-event type and waiting-time auxiliary task — **Defer until an event vocabulary passes**

**Novelty / equivalence.** This is a narrower form of prior generic next-row/first-move auxiliary ideas, not a new objective family. It is target-free self-supervision, but its value would come from a representation effect; the auxiliary classifier at inference does not itself help.

**Semantics / causality.** Future feature-event type/time may be an auxiliary training target, never a feature at time t. Censoring at sequence end and reset-local history are right. The event vocabulary cannot be interpreted as trade/queue state merely because a feature group changed.

**CPU route / support / falsifier.** A small head can be removed at inference; a large causal teacher could be distilled only after demonstrating that its state helps a compact student. Training remains nontrivial and the summary supplies no auxiliary predictability result. The cheapest no-fit falsifier is the card’s forward-chained feature-only renewal-vs-constant baseline. First exclude the near-always-changing extras; if a meaningful event family cannot beat its sequence-local base rate, defer the task.

### 11. Common/contrast target coordinates — **Defer as written**

**Novelty / equivalence.** An unrestricted full-rank rotation followed by inverse reconstruction is exactly a change of coordinates. It cannot change a model or score unless parameter sharing, regularization, or an optimization constraint changes. The card acknowledges this, but does not specify a unique constraint to test.

**Semantics / causality.** The target correlation is an offline outcome statistic, not an inference feature or a class-specific routing rule. The partial-law failure bars hard equality or exact decoding.

**CPU route / support / falsifier.** The change is cheap but has no compression benefit unless it reduces parameters. The reported negative dependence is strongest in the more extreme retrospective strata, so imposing one pooled relation may hurt the natural mixture. Before any fit, state the precise parameter tie/penalty and verify it is not algebraically equivalent to the existing two-head output; otherwise there is no experiment to run.

### 12. Clip-aware output geometry — **Defer until a frozen prediction artifact exists**

**Novelty / equivalence.** This is a narrow calibration transform, not a new model. Positive affine transformations do not change Pearson before clipping; only the scorer’s clipping boundary makes the proposal different. It must not reopen the already-negative pooled-correlation optimizer recipe.

**Semantics / causality.** It is legal as a deterministic final readout, but the scale/transform must be selected without confirmation outcomes and the official clip-weighted score must remain unchanged.

**CPU route / support / falsifier.** Runtime cost is negligible and no teacher is needed. The round correctly did not inspect predictions; there is no evidence that outputs approach ±2 or that the clipping boundary matters. Cheapest no-fit gate is just to inspect an already frozen, authorized prediction artifact’s saturation fraction and target weight mass. If none exists, keep deferred; do not create predictions to justify the idea.

## Three synthesis directions worth carrying forward

These are three distinct research paths, not a stack of twelve add-on features. The ranks reflect evidence-to-cost and legal implementation, not forecast gains.

1. **Reset-local soft innovation memory, without event/side labels.** Replace exact “event” triggers and guessed side signs with small, causal, robustly scaled residual summaries for stable feature groups, kept at a few fixed time scales. This absorbs the useful state-compression hypothesis from cards 1–3 and the continuous state question from 6–7 while avoiding named exchange semantics, hard regimes, and per-field clock proliferation. On CPU it is a few recurrence states; if a larger causal model later proves useful, distill its state/readout into those recurrences rather than serving the teacher. **Cheapest no-fit falsifier:** on fixed sequence halves, check whether robustly scaled group innovations have stable serial dependence or shift structure beyond a causal magnitude EWMA and row-step baseline. If not, no adaptive memory state is warranted. Current EDA does not establish this dependence.

2. **Low-rank causal cross-stream order kernel.** Test whether the ordering of i1 and i0 feature innovations adds information beyond totals, using a tiny fixed row-lag basis (optionally one antisymmetric degree-two term) and stable IDs, never adjacent-index depth assumptions. This absorbs and simplifies cards 4, 5, and 9 into one testable interaction; it does not reinstate the failed static i1 block. A larger causal teacher can be distilled to a few lag accumulators if the test passes. **Cheapest no-fit falsifier:** compare fixed-lag and area terms to count/first-sum-matched windows and within-sequence circular-shift controls, requiring direction to persist across both halves. The observed i1_volume drift is a robustness warning, not validation of the interaction.

3. **Causal sequence teacher to CPU student, with event-time pretraining only as a gated auxiliary.** If training is later authorized, the distinct delivery path is a high-capacity causal sequence teacher trained only on allowed training sequences, with a compact reset-local student matched to teacher states/predictions for CPU serving. A future-event family/interval objective may be included only after a feature-only renewal baseline establishes a nontrivial event vocabulary; it is never a serving input and never a target proxy. This differs from simply retaining the card-10 auxiliary head: the endpoint is a validated compressed causal state machine. **Cheapest no-fit falsifier:** first define the event vocabulary without extras and verify forward-segment renewal prediction beats a sequence-local constant baseline. There is no honest no-fit test of teacher/student forecast gain; that requires later authorized training and fixed-sequence evaluation, so no teacher experiment should be implied by this review.

All three directions require whole-sequence resets, warm-up/unscored updates where the contract requires them, no seq_ix feature, no future-input or oracle-label use, no external market data, and evaluation on the unchanged official clipped weighted-Pearson score. The four retrospective strata can organize descriptive checks only. Keep the prior field-age result, broad flow/volume negatives, chain-half volume-code reversal, and partial target-law failure visible in every later decision record.
