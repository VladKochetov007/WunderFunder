# ADR-0002: Separate active context from preserved evidence

Status: Accepted. Date: 2026-10-07.

## Context and evidence

The user requires no information loss and an extendable bandit tree. Long histories can also anchor agents to incorrect framing and stale low-value ideas.

## Decision

Maintain three layers: compact specs/live frontier; indexed hypothesis/experiment episodes; immutable raw evidence. Use ADRs for rationale. Retrieve source evidence progressively and give agents minimal role-specific packets. [Memory entry point](../README.md), [full methods](../../research/RESEARCH_METHODS.md).

## Alternatives and consequences

Putting all history into every prompt preserves accessibility but increases correlated framing. Deleting failures loses knowledge. Archive everything while removing unproductive recipes from active context; use fresh-context reframing and explicit reopen conditions.

## Revisit when

An agent cannot reconstruct a claim or repeated rediscovery indicates missing links/indexing.
