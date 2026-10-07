# ADR-0006: Repair execution failures within measured bounds

Status: Accepted. Date: 2026-10-07.

## Context and evidence

The laptop previously suffered OOM. The user authorizes Kaggle and autonomous repairs and wants computation results. E508 failed before training due to a [footer-range validator bug](../../research/campaign_2026-10-06/bandit_continuation/reviews/E508_FOOTER_RANGE_DIAGNOSIS.md). Infrastructure failure cannot determine scientific utility.

## Decision

Use verified local aggregatecgroup/GPU limits and separate cloud headroom. Profile actual shapes; observe resources, phase, exits andOOM counters; preserve failures and correct them under new IDs. Keep expected/observed input evidence before raising errors. Diagnose and rerun repaired attempts without routine user approval. Current numerical limits/allocation live in the [resumed contract](../../research/campaign_2026-10-06/bandit_continuation/RESUMED_ALLOCATION_20261007.md).

## Alternatives and consequences

Blind retry wastes compute and hides errors. Unlimited admission risks another OOM. Overgeneralized safety dependencies prevent results. Bounded diagnosis and arm-specific admission preserve both safety and progress. Explicit stops drain healthy work; explicit resumes preserve and supersede stop records.

## Revisit when

The user changes resources or budget, actual profiles demand a different execution shape, or monitoring cannot establish safe operation.
