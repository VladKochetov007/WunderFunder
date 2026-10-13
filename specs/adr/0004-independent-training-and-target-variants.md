# ADR-0004: Own training and explicit target variants

2026-10-13 successor: [ADR0035](0035-official-team-code-and-weight-sharing.md). Own-weights-only restriction superseded; explicit target variants and evaluation requirements remain.

Status: Accepted. Date: 2026-10-07.

## Context and evidence

The user confirms team merging is approved by hosts and requires all our models to be trained independently. The [replication audit](../../research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) found own teacher substitutions and an unregistered raw-versus-clipped family-target difference. The user wants combinations to reopen possibilities for weaker earlier recipes.

## Decision

Use our own trained weights. Declare recipe/ancestry deviations from the teammate reference. Preserve historical clipped-family configurations and create explicit matched raw-family variants. Main-target training changes require their own arms. Keep the official evaluation metric/clipping unchanged. [Execution specification](../experiment-execution.md).

## Alternatives and consequences

Silently changing clipping obscures attribution; assuming the same family names imply identical teachers overstates replication. Raw targets might help or harm, especially through tail gradients and representation transfer. Measure the variant and its combination rather than infer its effect.

## Revisit when

Matched evidence identifies a useful target treatment or new independently trained teacher families justify a declared replacement/interaction.
