# Microstructure round 1: independent ideas and source transfer limits

**Status:** independent first-round literature scout; proposals only. **Date:** 2026-10-17. No training, fitting, inference, submission, or new EDA was run for this note. The expected predictive gain of every proposal is unknown; none has an estimated score change.

## Contract and causal design boundary

Challenge 2 supplies 112 float32 features for two instruments in 20,000-row sequences. The data guide describes both scored targets as future movement of i0; their exact definitions are undisclosed. The latest read-only stream audit reports target-class changes on rows without observed i0 changes, possibly from float jitter or boundary effects still under investigation. No feature event is therefore treated as a label transition, and no exact target derivation is assumed. The official score clips labels and predictions to ±2, weights rows by the absolute clipped label, calculates pooled weighted Pearson separately for each target, then averages the two correlations. Each proposal below is an input-state or readout hypothesis that would still need to improve that unchanged score.

At serving time, every state is local to one sequence: reset on `seq_ix` change, update on warm-up and unscored rows, predict from current and earlier rows only, and never use `seq_ix` as an input or sequence-order key. No external market feed is assumed. The feature guide explicitly says component indices identify stable columns, not ordered depth levels. I therefore do not treat `p0…p10` or `v0…v10` as spatially adjacent or as best-to-deep levels. `i1` has no target. Any mention of an event time below means a row-step/event-count proxy; the release does not provide exchange timestamps.

The listed “no-fit diagnostics” are proposed deterministic descriptions of the supplied sequences. Features at row t must be built from rows ≤t. A target may appear only as a read-only outcome in an explicitly described offline diagnostic; it must never enter a feature, event state, or later prediction. No diagnostic below was run in this round.

## What the local history already says

The current local evidence changes what counts as a useful extension:

- The September microstructure catalog already proposed i1 flow accumulated between i0 price changes, refill-versus-depletion rates, event-clock flow surprise, a next-move auxiliary head, and post-move transition modeling. These are prior ideas, not fresh novelty.
- The audit records a broad 31-feature i0 order-flow block and a separate i1 cross-impact block without reproducible forecast gain. A static i1 block failing does not rule out a different temporal interaction, but it lowers that family's priority.
- A volume-code predictor was positive in-sample and negative under chain-half cross-fit (+0.0062 and −0.0078, respectively, in the audited probe). The discovered structure did not identify physical queue coordinates or portable residual signal.
- Per-field hold-age clocks were proposed on October 15; a read-only screen found extra clock resolution but did not establish forecast value. Discrete regime offsets/tables and several event heads also have negative or null scoped results. Existing HLOB/bilinear/TCN proposals are documented, but the audit did not establish that they were trained.
- The audit warns that target reconstruction and queue semantics remain partial. In particular, do not hard-code an inferred target identity or assume an event has “matured” after a chosen number of future events.

This is a duplicate screen against the reviewed local catalog and audit notes, not a claim that all historical branches or literature have been exhaustively searched. Each card names material overlap rather than claiming a family is new.

## Twelve hypothesis cards

### 1. Compact self- and mutual-excitation traces over observed event families

**Mechanism.** Define a small set of observable event families from exact changes in feature groups—for example i0 price-like, i0 volume-like, i0 trade, and the matching i1 groups. For each family maintain two or three causal exponential traces, incremented when that family changes. Feed counts and traces as a compact online state. A low-rank cross-excitation matrix can let a recent i1 or trade event transiently change another family’s event-rate context. Do not infer unobserved order types or actual aggressor signs from anonymous channels.

**Precedent and transfer.** Hawkes models capture event clustering and interactions in high-frequency finance; a 2024 paper forecasts bid/ask OFI distributions with a sum-of-exponentials Hawkes model (S03), while Cont et al. ground order-book-event flow in price formation (S01). Transfer assumption: exact group changes in this feed retain useful event-arrival dependence. OFI forecast quality in NSE tick data is not evidence of target lift in this dataset.

**Cheapest no-fit diagnostic.** Per sequence, compare each event family’s next-row conditional rate after recent same-family and cross-family events against a within-sequence circular-shift null, controlling only for elapsed row steps and sequence position. This reads feature events only.

**Falsifier.** Same- and cross-family rate contrasts vanish against the shift null, reverse inconsistently across sequence halves, or are already summarized by a single event-age counter. A later endpoint test would still be required even if clustering exists.

**Cost and history.** Low serving cost: O(number of families × number of traces) scalar updates. The direct OFI feature families have weak/negative local history; this compresses event clustering into an explicit causal state rather than adding another static flow block, but it remains low-priority until the descriptive gate passes.

### 2. Duration-surprise clock instead of raw elapsed rows

**Mechanism.** For a few aggregate event families, track the current elapsed row count since their latest event and divide it by a causal expected duration estimated only from that family’s completed past intervals in the current sequence. Use `log1p(elapsed / expected_duration)` or a bounded duration residual. This asks whether “waiting unusually long for this stream’s recent pace” differs from “waiting a long time in ordinary row units.” It is not 112 independent field clocks and does not assume true wall-clock durations.

**Precedent and transfer.** The ACD/ACM literature treats event intervals as stochastic state and shows that arrival timing can interact with the conditional distribution of marks (S02). Here event durations are only row-step proxies; the transfer does not assert that rows are uniformly sampled exchange time.

**Cheapest no-fit diagnostic.** Forward-scan each sequence and compare the empirical next-event frequency across bins of raw elapsed rows versus bins of elapsed / past expected duration. Estimate the expected duration from prior completed intervals only; never use the upcoming event to form its own feature.

**Falsifier.** The normalized clock gives no more stable event-rate separation than raw elapsed rows, or its scale estimate is too noisy within typical sequence lengths.

**Cost and history.** Low cost: a few event counters and duration summaries. This overlaps the old shared event-age and October per-field-age families. It is reasonable only as a much narrower rate-normalization test because the age screens established representation difference, not predictive value.

### 3. Power-law event memory for persistent signed-side proxies

**Mechanism.** Form a cautious side-activity proxy from the norm of recent innovations in the bid-side versus ask-side volume-like groups; do not call it executed order flow. Maintain a small logarithmic bank of causal traces or a truncated power-law convolution over this proxy, alongside the ordinary short exponential traces. This tests whether a long decaying memory captures persistent side activity that a handful of short clocks miss.

**Precedent and transfer.** Sato and Kanazawa’s 2023 analysis directly validates a power-law order-sign autocorrelation model on nine years of TSE account-level data (S04). They also describe compensation by size/liquidity dynamics. The challenge lacks trader identities and verified order signs, so only the general possibility of long memory transfers.

**Cheapest no-fit diagnostic.** On each sequence, compute autocorrelation of the side-activity proxy over log-spaced row and event-count lags. Compare its stability to exponential decay and to an in-sequence shuffled proxy; do not decode a volume dictionary or pool hidden states across sequences.

**Falsifier.** The proxy has no persistent sign or magnitude dependence beyond short lags, the power-law slope is unstable across within-sequence halves, or a few exponentials already summarize it.

**Cost and history.** Medium runtime cost for a small trace bank; low memory. Broad volume/flow feature screens and a cross-fit volume-code predictor were negative, so this is worth keeping only if a feature-only persistence signature survives without claiming physical queue semantics.

### 4. Directed cross-instrument lag surface in event time

**Mechanism.** Replace a static i1 feature block with a compact causal lag surface between i1 event families and i0 event families. Maintain counts/co-change marks over several preceding i1-event and row-step lags; expose direction separately for i1→i0 and i0→i1. A row-step clock is only an approximation because no timestamps are supplied. No i1-derived label or external feed is used.

**Precedent and transfer.** Cont, Cucuringu and Zhang find lagged cross-asset OFI helps forecast near-term returns in their multi-asset data, while contemporaneous cross-impact adds little after own multi-level OFI (S05). Hayashi–Yoshida’s estimator illustrates why asynchronous observation times need explicit treatment (S06); its estimator cannot be applied literally without the instruments’ timestamps.

**Cheapest no-fit diagnostic.** Compute within-sequence directed event cross-correlograms over a fixed, predeclared lag range, subtract a sequence-local circular-shift baseline, and report whether the direction and scale persist in both halves. Use feature events only.

**Falsifier.** No directed lag survives the shift baseline, lag direction changes by sequence half, or the signal is only contemporaneous. Any later predictive check must use causal lags and the unchanged full pooled score.

**Cost and history.** Medium cost. This is **not a new family**: it overlaps the September i1-event-flow proposal and the previously tested static i1 block. Its only candidate distinction is a lag-profile representation rather than cumulative i1 displacement since an i0 event. Retain it only if the feature-only lag diagnostic shows information the previous summaries did not test.

### 5. Low-order path signatures of the joint event stream

**Mechanism.** Treat a tiny set of normalized i0 price-like, i0 volume-like, i0 trade, and i1 event marks as a path with an explicit row-step/event-count time channel. Add only the first two signature levels over short causal windows. First-level terms measure accumulated movement; second-level antisymmetric areas distinguish, for example, “i1 changes then i0 changes” from the reverse ordering even when counts and totals match. Use streaming window updates and reset at sequence boundaries.

**Precedent and transfer.** Rough-path signatures have been used to represent path-dependent financial streams, including market behavior and execution traces (S10). The existing paper’s market tasks differ from this anonymized row stream; the transferable point is that a small set of iterated integrals can encode order of events, not that it will predict these labels.

**Cheapest no-fit diagnostic.** For a small fixed number of source-qualified sequences, compare second-level signature values for windows matched on first-level sums and event counts. Check whether the area terms survive an within-window event-order shuffle; do not select a basis using target outcomes.

**Falsifier.** Area terms are nearly zero, reconstructible from first-level sums, or unstable when window boundaries move by one event. A surviving descriptive term still needs a later fixed-endpoint comparison.

**Cost and history.** Medium cost for low-dimensional streams and degree two; state grows with window features. This representation is distinct from the old queue race, age, and static cross-impact summaries. It is an unusual but legal candidate because it uses only current/past feature order and no instrument identity beyond supplied i0/i1.

### 6. Continuous rolling covariance state rather than discrete regime labels

**Mechanism.** Maintain a causal exponentially weighted covariance over a handful of group-level innovation marks (e.g. i0 price, i0 volume, i1 price, i1 volume). Feed a continuous state such as trace, log-determinant, leading eigenvalue, and leading direction into the predictor. Keep it continuous; do not assign a hard “calm/busy” class or use one sequence to initialize another.

**Precedent and transfer.** A 2026 preprint examines LOB covariance-state geometry and argues that continuous state may generalize better than fine discrete regimes in its futures dataset (S11). It is a single recent preprint on 30–60-second aggregation, not confirmation for this 20,000-step release; the paper itself reports that finer discrete tiers fail to generalize.

**Cheapest no-fit diagnostic.** Per sequence, compare the covariance state’s dwell-time and variation to a scalar EWMA of total innovation magnitude. Determine whether the state carries a distinct continuous coordinate beyond activity scale. No targets or cross-sequence state are needed.

**Falsifier.** The covariance state collapses to total activity, changes too noisily to maintain a causal estimate, or its leading direction varies arbitrarily within one sequence.

**Cost and history.** Medium state-update cost for a 4×4 or similarly small matrix. Older regime tables/offsets were negative or null; this is a different continuous covariance representation and earns attention only if the feature-only redundancy check passes.

### 7. Online CUSUM alarm as a gate for a slow state (unconventional, cross-domain)

**Mechanism.** Borrow online process-monitoring methods: apply a two-sided CUSUM/FOCuS-style change detector to a causal, past-standardized innovation magnitude for each instrument group. The alarm may gate a separate slow state update or add a bounded “recent shift” scalar. It must not reset the main predictor, suppress predictions, or import calendars/news. Thresholds and normalizers must be fixed from past rows or training-only artifacts; state resets at every sequence boundary.

**Precedent and transfer.** Romano et al. develop FOCuS for high-frequency online change detection, with per-iteration computation logarithmic in stream length and tests on server anomalies (S12). Transfer assumption: abrupt changes in a normalized feature-innovation stream can be detected similarly. No claim that market regimes obey the server-data change model.

**Cheapest no-fit diagnostic.** Run the deterministic detector on feature traces only. Measure alarm rate on untouched sequence segments and detection delay on synthetic, injected level shifts in copies of those traces; do not inspect labels or tune on validation outcomes.

**Falsifier.** It alarms routinely on ordinary event clusters, misses the injected shifts at any useful threshold, or reduces to a noisy re-expression of event count. Later value must be shown with a fixed, causal endpoint comparison.

**Cost and history.** Low-to-medium serving cost, bounded state. This is the clearest external-domain mechanism in the round. Prior discrete regime-offset negatives argue against adding arbitrary regime tables; a shift alarm that only gates a slow state is a different, testable control mechanism.

### 8. Anonymous-component interaction graph without depth adjacency

**Mechanism.** Treat stable feature IDs as graph nodes and learn sparse pair interactions only among channels whose *past innovation* co-movement is reproducible in training sequences. Apply message passing on this graph while retaining each node’s identity and group. Never connect adjacent `p` or `v` indices merely because their names are adjacent; the guide does not give that meaning. The graph is a fixed training artifact, not hidden state shared across sequences.

**Precedent and transfer.** HLOB uses information-filtering graphs to capture nontrivial dependencies among LOB volume levels (S07). Its ordered, interpreted LOB structure is different from the challenge’s stable but anonymized component IDs. Local history contains HLOB/bilinear proposals but no confirmed training result.

**Cheapest no-fit diagnostic.** Within a few fixed sequences, compute channel-innovation co-change/correlation graphs in two disjoint time halves and compare edge stability to a within-sequence time-shuffle baseline. Do not build one graph from validation targets.

**Falsifier.** Candidate edges are unstable within sequences, vanish against the shuffle baseline, or depend on treating indices as depth order.

**Cost and history.** Medium-to-high model and artifact cost. It is an explicitly previously proposed architecture family, not a novelty claim. The narrow question is whether any stable, nonspatial graph structure exists in the actual components before authorizing a graph model.

### 9. State-first, flow-second interaction

**Mechanism.** Keep the current raw feature state as a full identity-preserving path and let a small flow/event branch contribute only through a gate conditioned on the current raw state. The test is whether a current-state representation changes how recent events matter, rather than whether generic flow features help unconditionally. Do not construct “best bid,” “queue depth,” or spread from feature indices unless a verified mapping exists.

**Precedent and transfer.** The 2025 MDQR paper models dependencies among queue states, market features, and order-size distributions (S08). It supports state-conditioned event modeling in a real futures book, but it is a simulator study with verified queue semantics. Challenge transfer is only the mechanism “event effects may depend on current observed state.”

**Cheapest no-fit diagnostic.** Use a fixed, simple robust summary of current raw i0/i1 group magnitudes and current event flags. Describe next observed feature-event rates by state-summary quantile; compare state-only and state-plus-event bins without fitting a predictive model. Keep all state causal.

**Falsifier.** Current-state bins do not change event rates, or event-rate changes are just total activity. If labels are later analyzed, use target only as outcome and retain full-row official metric.

**Cost and history.** Medium cost for a gated branch. It is related to the prior queue/refill and flow-surprise ideas, but specifically asks whether the incremental effect is state-conditional. Anonymous volume codes and the failed local flow families make this a low-priority diagnostic, not an assumed queue model.

### 10. Joint next-event type and waiting-time auxiliary state

**Mechanism.** Train a future research candidate with a lightweight auxiliary head that predicts the next *observable feature-event family* and its row-step waiting-time bucket from the current causal state. The auxiliary output can be dropped at serving; only its effect on the shared state would remain. Right-censor intervals at sequence end. Never use the competition’s future target values as auxiliary labels, and never feed the true next-event type/time into a forecast at row t.

**Precedent and transfer.** LOBDIF explicitly models a joint time/type distribution for future LOB events, though with a diffusion event generator (S09). This proposal is much smaller and uses observed feature changes; it borrows only the joint event-time/mark target.

**Cheapest no-fit diagnostic.** Measure whether a renewal baseline using only prior event intervals and family frequencies is detectably better than a per-sequence constant-rate/constant-mark baseline under forward-chained feature-only log loss. This is a descriptive diagnostic, not a training run.

**Falsifier.** Past state does not improve event-family or interval prediction over the sequence-local renewal baseline, or the auxiliary task learns only event counts.

**Cost and history.** Medium training cost, no required inference head. This is not wholly new: the October ledger lists generic next-row pretraining as queued/unverified, and an old first-move auxiliary head exists. The narrower event-type plus interarrival objective deserves consideration only if the no-fit renewal comparison shows useful structure; there is no local evidence of quality gain.

### 11. Full-rank common/contrast coordinates for the two targets

**Mechanism.** For analysis and a future training ablation, rotate the pair into `common=(t0+t1)/√2` and `contrast=(t0−t1)/√2`, but keep two unrestricted outputs and reconstruct the original pair before scoring. The transform is only a coordinate system: impose no exact target identity, class rule, or hard equality. A shared common representation plus a target-specific contrast readout is worth testing only if the target pair has stable dependence.

**Precedent and transfer.** This is primarily task-local: both targets describe future movement of i0 and the official scorer averages their separate correlations. The local audit finds target-law structure only partially reconstructed, so no external target formula transfers. The full-rank basis preserves the possible differences between targets.

**Cheapest no-fit diagnostic.** On supplied training labels, compute the pooled and per-sequence covariance/correlation of clipped `(t0,t1)` and of the common/contrast coordinates. Do not use the result as an inference feature; report sequence variation so one dominant sequence cannot dictate the conclusion.

**Falsifier.** Dependence is weak or changes sign across sequence subsets, or the proposed shared coordinate adds no stable structure. A full-rank rotation alone is mathematically equivalent; there must be a declared shared-parameter or regularization contrast in any later test.

**Cost and history.** Low-to-medium model change. Historical decoder/algebra claims are partial and exact label reconstruction did not meet broad-coverage requirements; this proposal does not hard-code them. It asks only whether the two scored outputs share a learnable component.

### 12. Clip-aware output geometry, not a new correlation loss

**Mechanism.** The scorer clips each prediction and target to ±2 and weights rows by the absolute clipped target. A fixed bounded output link or small target-specific saturation correction could matter only where the current forecast would clip. Explore the geometry as a readout transform; leave the official metric and model body unchanged. Any candidate transform must be frozen before inspecting its confirmation outcomes.

**Precedent and transfer.** The authoritative precedent is the local `global_wp_clip2_v1` scorer and Challenge 2 contract, not an external market claim. Weighted Pearson without clipping is invariant to positive affine output changes, so this idea has a narrow mechanism: only clipping and state-dependent saturation can distinguish readouts.

**Cheapest no-fit diagnostic.** On an existing frozen prediction artifact, if one is already approved for inspection, report (without searching or selecting a scale) the fraction of rows with `|p|>2`, the target weight mass at `|clip(y)|=2`, and each target’s contribution by those groups. This round did not read predictions. If there is no frozen prediction artifact available, defer the diagnostic.

**Falsifier.** Predictions do not approach the clipping boundary, the weighted target mass is not concentrated there, or a single preregistered smooth bounded transform leaves the official score unchanged on exposed development data.

**Cost and history.** Very low serving cost. A previous pooled-correlation objective recipe was negative, so do not reopen that optimizer family on this motivation. The proposed readout is only worth a fixed-procedure check if clipping is materially active; no score improvement is claimed.

## Suggested no-fit triage order

This is an information-per-cost order for deterministic diagnostics, not a model-quality ranking: (1) duration and event clustering (cards 1–2), (2) power-law persistence and directed lag profiles (3–4), (3) covariance and CUSUM state redundancy (6–7), (4) path-signature and graph stability (5, 8), then (5) the state-first, auxiliary-event, target-pair, and clipping diagnostics (9–12). Cards 4, 8, 10, and 11 have explicit local-history overlap and should not be described as novel.

Any later label association should use the existing exposure ledger, whole-sequence resets, chain/component grouping, and `global_wp_clip2_v1` unchanged. Validation panels are already exposed; EDA and resampling can prioritize ideas, but cannot establish an untouched holdout or justify a public-score forecast. No ideas here should be combined additively without a registered interaction test.

## Source register

See [`microstructure_sources.json`](microstructure_sources.json) for source verification dates, actual source content checked, transfer assumptions, and counterevidence. The source set contains twelve distinct primary works across order-flow impact, durations, point processes, long memory, asynchronous sampling, graph structure, event-stream prediction, rough paths, latent states, and online change detection. Recent works are treated as papers/preprints at the status visible on 2026-10-17; the source register does not imply they were peer reviewed unless stated.
