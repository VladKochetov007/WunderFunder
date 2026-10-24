# Event clocks, observed age, and asynchronous state: literature note

**Status:** source-grounded hypotheses, not results. No raw targets, fitting,
or new quality measurements were used. Public literature queries were generic.

**Correction record (2026-10-24).** The first draft proposed `log1p(age)` as a
new C2 encoding; source review found this is already implemented by the
canonical capped `normalized_age` path, so that card was a duplicate and is
replaced below. The first C3 wording could be read as H-396's frozen-state
replay; it is now explicitly a decaying event-state proposal, distinct from
both frozen-state replay and the existing held event clock. These corrections
preserve the earlier draft's history without treating either proposal as a
result.

## Local observation and limits

The no-target EDA reads 112 features plus sequence/scoring flags from 96
sampled validation row groups (1,920,000 sequence rows; 192,763 eligible
rows). It defines events as exact adjacent-row float32 changes, with row zero
excluded, and defines age in sampled rows since the last such change. It did
not read labels, targets, or predictions. The EDA's nine feature groups
separate `dp` and `dv`; the existing scientific grouped-age representation
has seven groups and combines those two blocks within each `i0`/`i1` family.

In this fixed sample, mean pairwise event Jaccard within `i0_p` and `i1_p` is
0.935 and 0.945; within each four-feature `dp` and `dv` block it is 1.000.
The `dp`-to-`dv` block means are 0.559 for `i0` and 0.773 for `i1`. These
describe co-change patterns only. A later root-owned scan of all 1,920,000
rows in this fixed sample found 65 exact event classes. Comparing each
selected scored field-age value with the seven-group representation, 14.392%
of values differ; the nine-family grouping differs on 11.533%. Thus the
groupings are not input-equivalent in this sample, and nine families are
closer to per-field ages by this descriptive measure. The result used no
targets and made zero fits. These percentages do not establish field meaning,
a physical clock, predictive value, or a quality advantage. See the
[geometry receipt](../../feasibility_scout/clock_eda/CLOCK_GEOMETRY_RESULT_R1.json).
This sample is exposed validation data, not a fresh holdout.

## Primary sources and transfer limits

1. Kaul, Yates, and Gruteser, [“Real-Time Status: How Often Should One
Update?”](https://winlab.rutgers.edu/~gruteser/papers/sanjitnew.pdf), INFOCOM
2012. The paper defines status age from current time and the generation time
of the latest received update, and shows that freshness is distinct from
minimizing delay or maximizing update rate. This motivates testing an age
representation separately from event frequency. Its model assumes
timestamped source updates and a delivery/queueing system; the EDA here uses
row-count age and has no verified physical timestamps, so the analogy is
limited.
2. Akidau et al., [“The Dataflow Model”](https://www.vldb.org/pvldb/vol8/p1792-Akidau.pdf),
PVLDB 8(12), 2015. It treats event time and processing/observation time as
distinct time domains. The transferable design lesson is to retain clock
coordinates separately until their relationship is justified. This is a
stream-processing semantics paper, not a predictive-model result; no
out-of-order or timestamp semantics have been established for these rows.
3. Neil, Pfeiffer, and Liu, [“Phased LSTM: Accelerating Recurrent Network
Training for Long or Event-based Sequences”](https://arxiv.org/abs/1610.09513),
2016. It adds a time gate for irregular asynchronous streams and inputs from
different update rates. This suggests event-conditioned state updates as a
transferable mechanism. Its gate uses sampling time; a feature not changing
between adjacent rows here does not prove that a recurrent update is
unnecessary.
4. Mei and Eisner, [“The Neural Hawkes Process: A Neurally Self-Modulating
Multivariate Point Process”](https://arxiv.org/abs/1612.09328), NeurIPS 2017.
It models marked events in continuous time with a recurrent hidden state that
evolves between events. This motivates a jump-plus-decay state as an
unconventional transfer. It is a generative event-time model; neither its
objective nor its data assumptions establish benefit for this supervised,
row-indexed task.

## Hypothesis cards

All comparisons keep the official metric, labels, split membership, model
ancestry, and strict FP32 recipe fixed. The paired cells below are examples
from the prior source-qualified short-screen convention; the root must pin
the exact compatible recipe and fresh source-qualified folds/seeds before
admission. No historical control is assumed reusable: a control may be reused
only after checkpoint, prediction, fitting-membership, and recipe provenance
are verified.

### C1 — Keep separate `dp` and `dv` ages

**Mechanism.** Feed the nine group-age channels rather than merging `dp` and
`dv` into seven. The hypothesis is that different recency histories may carry
complementary information even when within-block change events are highly
similar.

**Input-level evidence.** The completed sample-wide comparison shows the
seven- and nine-group age tensors differ at scored rows; the representation
equivalence falsifier did not trigger. The nine-family grouping also has a
smaller field-age mismatch in this sample, but that is not quality evidence.

**Smallest falsifier.** Run two matched fold/seed cells (four fits with fresh
controls) using the same age scale, source, model and recipe. If nine groups
do not improve the official endpoint consistently across those pairs, retain
the seven-group model at this fidelity; that does not reject other age
representations or scales.

### C2 — Scale age by past-only local event activity

**Mechanism.** The canonical
[`normalized_age`](../../../../../src/wnn/training/age_readout_features.py)
already uses reset-local, capped `log1p(age)` scaled to `[0,1]` through
[`research_features.py`](../../../../../src/wnn/research_features.py); repeating
that transform is not a new arm. A materially different candidate is to
multiply each existing seven-group row-age by a past-only estimate of that
group's change rate, then apply the same capped `log1p` scale. The rate
estimate would use only prior input change events and reset with the sequence.
This encodes how stale an unchanged span is relative to recent activity,
without claiming physical time or importing target information. Its
half-life/initialization must be frozen before fitting. This may be a narrow
implementation variant of prior
event-rate/clock work, not an independent new mechanism: the repository
already proposes comparing event-clock and row-clock summaries in
[H-ASTRA23-02-B](../../../../../research/unified/reviews/astra_ultra_audit_20261023_r1/03_memory_and_time.md),
and earlier review explicitly asks whether error tracks row distance, event
count, or their disagreement. Dedupe it against that lineage before creating
a ledger arm.

**Smallest falsifier.** Verify prefix-causality, reset behavior, and that the
transformed inputs are not identical to the existing seven-group age channel
on the source-qualified fitting panel. One matched seven-group current-age
control/treatment cell (two fits if the control is fresh) is the smallest
model falsifier; it cannot establish a reliable quality direction. A second
matched cell is needed for a minimal repeated screen.

### C3 — Event-triggered jump with between-event decay

**Mechanism.** As an unconventional transfer from continuous-time event
models, retain an always-updated row-time path and add a causal event-state
sidecar. Update the sidecar at a predeclared group event, and decay its state
between events as a function of elapsed rows; keep the output path active on
every row. This differs from the existing
[`DualClockModel`](../../../../../src/wnn/nn.py), whose event-GRU state is held
between selected events and receives `log1p(gap)` at its next event, and from
[H-396](../../../../../research/unified/reviews/H396_EVENT_TRIGGERED_GRU_FEASIBILITY_20261023_R1.md),
which proposes holding the *row-GRU* state on a novelty trigger while
rebuilding current features/projection/readout. The proposed between-event
decay must be reviewed against other event-clock lineages before it can be
called novel. No trainer copy or code change is proposed here.

**Smallest falsifier.** One matched control/treatment cell (two fits if the
control is fresh), with the row-time path and event definition fixed. Reject
the implementation if its event state is not prefix-causal, leaks across
sequence resets, or changes label/update timing. One pair is only a screening
observation and is not a reliable quality estimate.

## Proposed first screen: six new scientific fits maximum

Use two matched cells, for example F0/seed1201 and F2/seed1202, subject to
source qualification. Train two fresh baseline controls, then pair them with
two C1 treatments, one C2 treatment on the first cell, and one C3 treatment
on the second: **2 controls + 2 C1 + 1 C2 + 1 C3 = 6 new fits total**.
Each treatment reuses only its exact same-cell control, giving four paired
contrasts. There are no reused historical controls in this count. C1 gets two
cells; C2 and C3 each get one and therefore remain highly uncertain. This
small panel cannot isolate fold effects from seed effects or establish
significance. If full matched cells for all three mechanisms are required,
that is eight fits (two controls plus two treatments per mechanism), above
the initial screen cap. No runtime cost is estimated here; profile and admit
only after the root has checked actual throughput and source closure.

This note proposes tests; it reports no new score, significance, scientific
quality grade, or public-score implication. The three cards are not claimed
to be three independent mechanisms; C2 and C3 need lineage deduplication
before any new experiment record.
