# Representation extensions after the clock and event probes

Controller proposals, 2026-09-30. These are proposals for criticism, not approved
fits, statistical gains or public forecasts. No new GPU expenditure is committed.
The existing frozen spread comparison must finish first. The independent clock
replay, optimized D1 cost measurement and execution access remain prerequisites.

## Evidence motivating a change

E282 finds moderate extra-feature activity associations, with no simple event-rate
identity. Extra fields can change while all104 published market fields remain
unchanged. Hidden source inputs or repeated identical messages remain possible.
E283 finds identical conditional ordering of a0,a5,a6 immediately after joint
lower-endpoint observations on16 fresh TRAIN components. This suggests an interval
coordinate but does not identify units, additive time or the numeric inverse.

The earlier inverse-clock fit failed its noncollapse gate; that failure remains.
The independent optimized D1 numerical proof passes, but a positive linear kernel
stores compressed weighted event summaries. It does not implement an explicit
32-event softmax ring. These distinctions motivate two separate mechanisms.

## C1: interval-anchored additive coordinate, then variable-decay state

**Representation:** treat a0 as a candidate observed increment coordinate. Fit
positive monotone f0 and increasing age maps f5,f6 on TRAIN inputs only. On
conservative clock-quiet transitions test

```
f5(a5[t]) - f5(a5[t-1]) = f0(a0[t])
f6(a6[t]) - f6(a6[t-1]) = f0(a0[t])
```

Unlike E274, this imposes agreement with a third observed input rather than only
agreement between two fitted differences. That adds constraints; it does not
automatically prevent collapse. A new frozen protocol must retain noncollapse,
supported-knot, unsupported-mass, component-mass and heldout-direction gates.
First test the input equations, with positive row-count, CDF and lognormal interval
controls, before a neural fit. Report lower-endpoint ambiguity and unknown clock
resets. Learn no target-dependent clock, and claim no seconds.

**Model if the input proof survives:** use the identified positive interval to
control stable scalar state decay, with raw inputs and an unknown flag retained.
Compare against the same active state and parameter tensors with unit row duration.
An additional independently learned positive a0 duration is a separate mechanism,
not evidence that the recovered coordinate is correct. Variable decay needs a
bounded causal scan; the fixed-coefficient FFT implementation cannot be reused
unchanged. Long-sequence, incoming-state, chunk, gradient and CPU/native numeric
checks precede any fitting.

**Falsifier:** failed additive equations on unseen components, collapsed mass,
unsupported extrapolation, or no exact Global WP win against the matched row-time
model. A clock proof alone earns no predictive-quality claim.

## C2: explicit finite event ring with softmax retrieval

**Representation:** maintain the last32 causal a6-endpoint event embeddings, with
sequence-local state, and retrieve them using a query from the current GRU state.
The write flag is the raw input endpoint, with existing origin/quiet exceptions
handled explicitly. It is not a target-derived movement flag. The ring stores
input embeddings or causal hidden states, never labels, residuals or mask bits.

```
query = Wq * current_hidden
key[j] = Wk * event_embedding[j]
value[j] = Wv * event_embedding[j]
weights = softmax(query @ key / sqrt(d) + fixed_distance_bias)
prediction = base_prediction + residual_head(sum(weights[j] * value[j]))
```

Freeze rank, value width, distance bias and zero-initialized residual output before
training. A softmax ring can retain a rare event without reducing it to the small
set of moments held by a linear kernel; whether this helps these targets is unknown.
No claimed gain follows from that capacity argument.

**Matched controls:** replace content keys with position encodings of the same
width, with the same active learned key/query/value projections and number of
stored values. Include a separately budgeted current-state/last-event additive
head. Parameters that do not affect a control's output cannot count toward matching.
Disclose different operation counts or use equal measured budgets; do not hide
cheaper cached position projections. Ring size32 is fixed, not a post-score sweep.

**Cheap implementation falsifier:** fixed TRAIN feature-only exposure establishes
nonempty rings, support for32-event histories, and variation beyond the last event.
Synthetic stream versus vectorized gather must agree for warmup, sequence reset,
chunk boundaries, empty/partial/full rings and20,000-row paths. Padding never
writes. Current event inclusion/exclusion must be frozen consistently. A short
batch32/chunk2000 GPU profile must fit5GB and the assigned whole-fit cap before any
target experiment; no smaller batch substitution.

**Quality falsifier:** no fixed-blend gain >=.0005 against the clean reference, or
no gain over the position-only kernel on the identical complete row/mask set.
Confirmation, class floor, mask sensitivity and public expectation gates apply
unchanged. A learned attention picture or sign AUC is not the score gate.

## Execution order and publication boundary

1. Finish and bind the already-frozen spread comparison without inspecting partial
   quality, preserving original-plus-resume resource accounting.
2. Collect independently reproduced feature relations and optimized D1 cost.
3. Critic decides whether C1 or C2 has the stronger cheap falsifier and budget case.
4. Only then freeze a runnable candidate/control evaluator and fit within the
   remaining campaign allowance and final-fitting reserve.

Model fitting uses supplied competition data only. External data can supply
generator counterexamples and distribution comparisons, but no external fitted
numeric state or examples are proposed for submission training. The public
incumbent remains0.6622; neither proposal has a calibrated expected public gain.
