# ADR0013: bounded paid compute and campaign-owned instances only

Status: accepted, 2026-10-07. Basis: user approved the continuous research plan, chose evidence-led Vast spending, and explicitly restricted actions to machines rented by this campaign.

Spending-stage gates superseded on 2026-10-08 by [ADR0015](0015-flexible-credit-bounded-parallel-compute.md). Preserve this original allocation and all frozen receipts; ownership and bounded execution remain binding.

## Decision

Use a finite queue across local GPU, Kaggle GPU/TPU and Vast, with a reviewed successor ready before a running job ends. Existing local guards remain authoritative; cloud jobs use separate measured headroom. Source/metric/ancestry invariants remain unchanged. Allocation reflects long-term component and interaction value, not only standalone score.

Authorize at most $4 all-in for the first Vast cohort and $20 cumulatively, including storage, transfer and cleanup. Release later cohorts only from reviewed completed evidence. No subscription or credit top-up. This supersedes the earlier no-paid-rental restriction within this exact scope.

Only instances provably created by this campaign may be connected to, collected from, modified, stopped or destroyed. Require a durable creation receipt bound to attempt, account scope, unique campaign label and provider instance ID. Do not adopt pre-existing instances or rely on labels alone. Persist create intent before contacting the provider; ambiguous outcomes require reconciliation and never blind retry. Do not perform bulk cleanup or host-level actions affecting other tenants.

Paid work requires a frozen runnable recipe, current all-in quote, bounded lifetime, artifact collection and ownership-checked teardown. Monitor only outstanding campaign work; a provider terminal state without verified artifacts is incomplete. An empty useful paid queue triggers collection and destruction of its owned instance. Keep the API credential outside logs, model prompts and workers.

## Consequences and validation

The controller ledger must preserve budget reservations across crashes, block duplicate launch after an uncertain submit, and reject unrelated instance IDs before a network mutation. Tests cover those cases, missing/mismatched artifacts and unverified teardown. Resource failures remain execution evidence and do not falsify scientific mechanisms.

The [saved implementation plan](../plans/2026-10-07-continuous-research.md) defines the initial hypotheses, placements, evidence-triggered descendants and resource ceilings. All actual outcomes stay in the append-only experiment records and original lineage artifacts.
