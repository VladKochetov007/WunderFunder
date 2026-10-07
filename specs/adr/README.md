# Architecture and research decision records

ADRs explain why a consequential research or orchestration decision was made. They are not claims that its hypothesis worked. Link the actual evidence and alternative costs; declare consequences and conditions for revisiting it.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-long-term-winning.md) | Optimize long-term winning with exploration and deepening | Accepted |
| [0002](0002-layered-evidence-memory.md) | Keep compact specifications, indexed episodes and immutable evidence | Accepted |
| [0003](0003-independent-diverse-ideation.md) | Use diverse agents and random/directed literature components | Accepted |
| [0004](0004-independent-training-and-target-variants.md) | Train own models; explicitly distinguish raw/clipped target recipes | Accepted |
| [0005](0005-result-ownership-and-independent-arms.md) | Assign end-to-end result owners and arm-specific dependencies | Accepted |
| [0006](0006-resource-safe-autonomous-execution.md) | Bound resources and autonomously repair identified execution failures | Accepted |
| [0007](0007-specialized-research-skills.md) | Specialize research workflows around concrete failure modes | Accepted |
| [0008](0008-research-scale-beyond-serving-envelope.md) | Pursue top-1/>0.700 with training scale beyond submission limits when rules permit | Accepted |
| [0009](0009-uncertainty-is-not-a-kill-switch.md) | Treat uncertainty as evidence, not a universal stop rule | Accepted |
| [0010](0010-continuous-experiment-delivery.md) | Keep authorized experiment owners active through repairs and measured results | Accepted |

For a new decision, copy [the template](template.md), choose the next unused number and add it here. If superseding a decision, preserve its text/history and link both records. User instructions take precedence; record their effective scope rather than quietly editing an old scientific specification.

- [ADR0011: measured pipeline utilization](0011-measured-pipeline-utilization.md): verifieddigestreuse, completecohortcost and bounded loaderprofiling.
- [ADR0012: independently implement teammate methods](0012-independent-teammate-method-implementation.md): read code for ideas and implementation details; independently write training and serving code in our own project conventions.
- [ADR0013: bounded paid compute and owned instances](0013-owned-paid-compute-and-continuous-queue.md): finite useful-run queues, $4 initial/$20 cumulative Vast spend, and no actions on pre-existing or unrelated machines.
- [ADR0014: continuous useful compute](0014-continuous-useful-compute.md): keep local/Kaggle/authorized Vast lanes supplied with useful work until explicit user stop; own repairs, terminal artifacts and prepared successors.
- [ADR0015: flexible credit-bounded parallel compute](0015-flexible-credit-bounded-parallel-compute.md): latest user spending instruction removes the first-cohort release gate, retaining existing credit and a conservative $20 cumulative operating ceiling with reviewed bounded cohorts.
