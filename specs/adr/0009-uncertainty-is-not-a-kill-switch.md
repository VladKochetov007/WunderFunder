# ADR-0009: Treat uncertainty as evidence, not a universal stop rule

Status: Accepted. Date: 2026-10-07.

## Context and evidence

The user observed that requiring a confidence interval to exclude zero can reject
good research ideas, especially when evaluation units are few, expensive, or
already exposed. E-511's C1 pair has a positive point difference of +0.001023
against C0, with a connected-component interval spanning -0.001473 to +0.004619
on exposed fold 0. That result is unresolved; it is not evidence that the
teacher-target idea has no value. E-518 screens repeated the same fixed E-514
candidate pool, and its registered endpoint slightly trailed its controls.
Those findings support stopping those exact pool/rule descendants while
retaining the broader complementarity family.

## Decision

An interval crossing zero describes uncertainty under its sampling and
evaluation assumptions. It is not, by itself, a reason to reject a hypothesis,
close a lineage, or deny a bounded experiment. Classify evidence separately
from allocation:

- **Validity and resource safety are hard gates.** Data identity, recipe
  semantics, causal serving, competition rules and OOM-safe resource controls
  must pass before a run. Scientific optimism cannot waive them.
- **Scientific evidence is graded.** Record the effect estimate, uncertainty,
  exposure, training variation, control quality and whether the result tests
  the stated mechanism. A zero-crossing interval is “unresolved” unless its
  width and design support a more specific practical-equivalence conclusion.
- **Allocation is cost- and value-of-information-aware.** Consider mechanism
  plausibility, expected gain, cost, reversibility, information gained,
  reusable artifacts, and opportunity cost. Keep bounded exploration across
  independent families even when leading arms remain uncertain. The older
  scalar knowledge-gradient script assumes standalone, additive system gains;
  do not use that assumption to screen conditional components. Represent
  potential synergy as an explicit interaction and evaluate its complete
  endpoint.
- **Stopping is claim-specific.** A failed implementation closes no scientific
  claim. Repeated failure can stop an exact recipe or spending path when the
  evidence is relevant, the experiment had power to discriminate useful
  outcomes, or the opportunity cost dominates. Preserve broader descendants
  unless their mechanism was actually tested.
- **Promote in stages.** Positive but uncertain effects may earn another
  discriminating or independent test; they do not become public claims or
  additive score forecasts. Previously exposed groups and dependent arms do
  not become fresh evidence through new seeds, agents or intervals.

Use confidence intervals and other uncertainty summaries to decide what the
next test can learn, not as universal accept/reject thresholds. The research
methods document is authoritative for the operational decision process.

## Alternatives and consequences

A fixed confidence-level gate is simple, but it can make costly or small-sample
research permanently inconclusive and underfund novel mechanisms. Ignoring
uncertainty entirely would over-promote noise. The chosen approach preserves
uncertainty as one input, uses inexpensive discriminating work when its value
justifies the cost, and retains hard execution-safety and evidence-validity
gates. It may spend some bounded resources on ideas that later fail; those runs
must still produce interpretable results and reusable artifacts.

## Revisit when

Revisit if the user changes risk preference, repeated low-information tests
consume the exploration reserve, or prospective records show that this policy
systematically over- or under-allocates to uncertain arms.
