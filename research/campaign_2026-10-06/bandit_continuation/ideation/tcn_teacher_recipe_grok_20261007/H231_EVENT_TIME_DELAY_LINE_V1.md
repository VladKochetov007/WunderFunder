# H231: causal event-time slow delay line

Proposer: native Grok CLI/grok-4.7, source-bound round4. Public proposal and source hashes are preserved in ROUND4_PUBLIC_STDOUT.txt and SOURCE_TASK_RECEIPT_V4.json. This is a component proposal, not a measured result.

Mechanism: retain a fast row-time branch, but advance a slow causal delay line only when an input-derived quote-change predicate fires. A fixed event-capacity state can cover variable row spans, preserving event memory through publication bursts. Existing TCN activation tails and TwoClockModel provide reusable components. No premise that labels stay constant on quiet rows is needed or established.

Cheap first test: an explicit toy event mask and causal streaming state; require full/event-chunk parity, exact holds on quiet rows, correct updates on selected rows, future-prefix invariance and independent sequence reset. Check boundary duplicates/missing events, and report event count versus row span without fitting targets. Zero future/label-derived predicates.

Matched candidate: row-time fast branch plus event-time slow branch versus equal-capacity row-clock branch with identical causal input/output alignment. Separate the quote-change predicate from the a6-endpoint predicate as named variants. Merely feeding fewer rows is not the same as holding a slow latent branch. Keep scalar counters/state bounded and deterministic at serving.

Evidence endpoint: separately admit measured actual-device shape/profile and own fit; then test fixed student/blend contribution, including a individually weaker teacher if complementary. Use known component units for descriptive excluded-fold resampling, retain exposure and never infer a public score by adding a local delta. A bad earlier distilled TCN does not settle this representation change; preserve its history.

Resources: no fit, GPU profile or production implementation admitted here. Numerical priors and expected public gain remain null. Priority follows complete useful experiment cost and independence from current wide-teacher lineage, not novelty wording.
