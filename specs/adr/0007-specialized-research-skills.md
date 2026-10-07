# ADR-0007: Specialize research skills around failure modes

Status: Accepted. Date: 2026-10-07.

## Context and evidence

The user requested the specialist-skill pattern used in `/home/vlad/development/MFTE/.agents/skills/`, adapted to research. Inspected `independent-review`, `unsafe-concurrency`, `rust-change`, `performance-evidence`, `parallel-worktrees` and `architecture-decision`. Their useful pattern is a concrete contract, traced behavior, independent checks and an evidence handoff, rather than generic expert personas.

Our replication reflection exposed recipe drift, code-only delegation without a completed scientific result, coupled prerequisites and overinterpretation risks. The resumed critic caught an actual profile-command conflict; narrow checks can improve throughput without becoming an infrastructure project.

## Decision

Keep `conduct-creative-research` as the orchestrator. Add four focused project skills: `research-recipe-audit`, `research-evidence-review`, `research-experiment-delivery` and `research-reframe`. Route them by actual task/failure mode and load only necessary guidance. Keep current scores, budgets and model choices in project specs/contracts, rather than bake them into reusable skill rules.

## Alternatives and consequences

One giant skill makes every task carry all protocol details; a large persona swarm adds coordination without independent evidence. Four scoped workflows make contracts and completion checks explicit. The reviewer must not self-approve, and specialized review is needed for material research claims/changes, not every harmless edit. No copied MFTE Rust-specific invariants or automatic permission flows apply here.

## Revisit when

Repeated failure suggests a missing specialist, overlapping skills cause duplicate gates, or unnecessary review delays runnable experiments. Evaluate workflow changes by actual correctness, completed results, costs and information gained.
