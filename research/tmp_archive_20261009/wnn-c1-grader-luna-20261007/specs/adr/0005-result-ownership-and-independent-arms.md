# ADR-0005: Own complete results and separate dependencies

Status: Accepted. Date:2026-10-07.

## Context and evidence

The prior [replication attempt](../../research/campaign_2026-10-06/bandit_continuation/reviews/MAXIM_REPLICATION_REFLECTION.md) produced C0 and code, but no heterogeneous student fit. Builders owned code, controller owned execution, and even the simple homogeneous control depended on the missing full-family cache/cloud baseline.

## Decision

Give a lead actor an end-to-end trained-model/prediction deliverable coordinated with controller admission. Give cloud operator terminal execution/artifact ownership and grader independent fixed-comparison ownership. Require only arm-specific dependencies. Obtain the leading result before broadening all ablations. [Agent contract](../agent-contract.md).

## Alternatives and consequences

Code-only delegation is useful when named explicitly but leaves scientific completion to another owner. A monolithic prerequisite chain prevents independently runnable experiments. Independent arms improve throughput while retaining necessary data/resource checks.

## Revisit when

Actual dependencies require coordinated staging or a task's scope is explicitly code/review-only.
