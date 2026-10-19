# TradeFM-inspired event dynamics, adapted to anonymous causal rows

User-supplied source: [TradeFM,2602.23784](https://arxiv.org/abs/2602.23784),
[primary PDF](https://arxiv.org/pdf/2602.23784), inspected October19,2026.

## Relevant source evidence

TradeFM uses a524.4M decoder with grouped feature embeddings, RoPE/GQA and
discretized event prediction. Its data contain real trade fields and timestamps.
Appendix B reports context 1024 and FP16 AdamW training on 10.7B tokens. Its
generation evaluation concerns market distribution properties, not our official
weighted-Pearson regression or GRU distillation. Those are external results,
not local evidence. We did not download its data, weights or implementation.

## Proposed local mechanism

Learn a predictive state of observable transitions while retaining per-row raw
inputs and supervised target outputs. This makes the representation responsible
for future changes rather than only reconstructing persistent values. It is a
new local adaptation, not a reproduction of the source's trading simulator.

1. Retain raw 112+event 99 row inputs. The same candidate/control encoder may
   include fitting-only numerical bins, but choose that basis before comparing
   auxiliary losses; do not change tokenization and objective simultaneously.
2. Define group changes using canonical past/current feature-update semantics.
   Use schema groups; do not label inferred physical buy/sell/add/cancel events.
   Report fitting-side change frequencies before choosing predictable groups.
   A high-innovation threshold is an alternative definition requiring its own
   frozen fitting-only calibration; do not replace exact changes invisibly.
3. For each anchor t, predict the next observable change within a fixed H=64
   rows, with a later H=256 arm if useful. Use hazards at each future offset,
   with an optional coarser tail after the near offsets, and separate group,
   direction and magnitude-bin heads. For the first implementation, all heads
   read h[t] directly; no future mark or event time enters the backbone/readout.
4. Construct future marks from supplied fitting observations only. Quantile edges
   for signed changes are fitted on fitting components and frozen. Include
   explicit tail bins, preserve the original raw inputs and avoid a Cartesian
   vocabulary over all 112 features.
5. Let z[t,j] be the event hazard conditional on no earlier event. If the next
   event occurs at offset k<=H, the timing NLL is
   -sum_(j<k) log(1-z[t,j]) -log(z[t,k]). If no event occurs over the observed H,
   it is -sum_(j<=H) log(1-z[t,j]). A sequence ending after only c<H observed
   offsets contributes survival terms through c, with no invented event/tail
   label afterward. Zero observed future offsets contribute no timing term.
   Use stable log-sigmoid arithmetic. Never cross sequence or excluded-component
   boundaries; a crop must supply a target-only future tail or censor its loss.
6. Mark losses apply only to an observed event within the declared horizon.
   Quiet rows still contribute timing likelihood and the main supervised loss.
   Normalize timing and mark losses separately over valid anchors/events, and
   each group's support explicitly. Apply the same global-denominator DDP rules
   as the supervised term; do not let variable event frequency silently scale it.

The initial joint objective is L_sup +0.05*(L_timing+L_marks), after separate
normalization. This coefficient is a proposal. Inspect shared-backbone gradient
ratios during bounded engineering work, then freeze it before scientific fitting.
Both arms instantiate the same heads; the control sets the auxiliary coefficient
to zero, with graph participation handled consistently in DDP. The auxiliary
targets are never inputs. This is distinct from feedback of model predictions.

No simulator or sampled rollout is required at serving. The teacher reads the
observed prefix, produces deterministic target predictions and optionally
predictive-state/mark distributions for distillation. Preserve every quiet row
in evaluation; an event-only model evaluated only on events answers the wrong
competition question. Online fitting to true labels is still unavailable because
their maturity horizon is unknown.

## What we should not transfer without evidence

Our columns are anonymous and may already be transformed. Real-unit log-volume,
mid-price ratios or VWAP formulas are not justified merely by a price-like name.
There are no supplied wall-clock intervals or asset identifiers. Observable
row-age/event-count clocks are valid alternatives. Fitting-only scalar bases and
causal residual/scale channels are experiments, not a claim of market invariance.
Never use whole-sequence centering/scaling for early predictions.

Paper dataset-token ratios do not set our optimal teacher size: our fixed rows
are correlated and tokenization differs. Its FP16 recipe does not establish strict
FP32 backend support, Muon benefit or a B300 ETA. More realistic generated series
would not itself establish better competition targets, and generated series
would require a separate data/provenance and target-validity experiment.

## Concrete next contrast and accounting

[Event-objective recipe](transition_recipe.json): same 0.68B teacher, supervised
versus supervised+event likelihood, at two matched fold/seed cells =four TOTAL
new teacher fits. This is a high-priority dynamics comparison BEFORE extending
the full scale ladder, or alongside it only under a separate device/time
allocation. The capacity comparison in recipe.json is another four-fit panel,
not four free extra fits. Testing both objectives at both scales and both cells
would cost eight teacher fits, before students.

Preserve per-cell official target scores, hazard calibration/event-mark NLL and
change/no-change/sequence-age diagnostics. Better NLL alone does not promote the
teacher. If target/representation evidence is useful, use existing clean teacher
caches for a separate matched GRU KD comparison. Start with final target regression
KD; auxiliary predictive-state transfer is its own connection hypothesis. A
negative first target delta can still justify that bounded transfer test when a
specific representation mechanism supports it.

H-OBSERVABLE-EVENT-DYNAMICS-20261019-R1 and
E-OBSERVABLE-EVENT-DYNAMICS-20261019-R1 register this proposal. Independent Luna/max
scout reviewed the paper and recommended observable-change survival learning,
while warning about true-unit features, precision, sampling and metric mismatch.
The controller selected canonical event definitions over the scout's thresholded
large-innovation definition for the initial arm, avoiding an extra fitted event
selection mechanism. No implementation, fit, likelihood result or student gain.
