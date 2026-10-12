# ADR0034: Resume strategy experiments with the full GPU allowance

Status: Accepted. Date:2026-10-13 Europe/Kyiv (2026-10-12T22:59:14Z).

## Authority and context

The user requested implementation and deep experiments for every idea in the October12 public/private strategy review, selected a maximum .001 ordinary-CV sacrifice for private robustness, authorized "all the kaggle allowance and local gpu without limitations", and then said "Implement the plan."

## Decision

Resume research, preserving the October12 stop as a historical record. Use all current Kaggle GPU/TPU allowance and subsequent renewals while this campaign is active. Supersede inherited Kaggle campaign sub-ceilings and local GPU wall-time/5GB observed/4.3GB allocator limits. Size jobs against actual available physical GPU memory with measured operational headroom; do not use OOM to discover fit. Keep the single local GPU-worker admission authority.

The separate5CPU,17GB committed target,18GB memory-high/20GB hard RAM and no-swap rules remain. CPU is analysis/support, not large model fitting. No new paid rental, top-up or billing change is authorized. Native subordinate roles remain Luna/max, at most three; no new Astra/Sol exception.

Implement the approved [execution plan](../../research/campaign_2026-10-13/strategy_execution_r1/PLAN.md), with [contract](../../research/campaign_2026-10-13/strategy_execution_r1/CONTRACT.json) and [coverage](../../research/campaign_2026-10-13/strategy_execution_r1/COVERAGE.json). Do not equate code completion with an experiment. Use exposed F0/F3 paired screens, broader fold/seed endpoint checks, explicit teacher ancestry, independent review and unchanged official metric. Rank private candidates within.001 of best mean CV by predeclared relative stress robustness.

## Compatibility and consequences

Preserve frozen recipes, source snapshots, raw evidence and archived submissions. Apply changed resource authority to newly admitted jobs; no stale watchdog may kill a healthy job at a withdrawn GPU threshold. Update the current admission path narrowly and independently test it before reopening. Reuse completed teacher/cache artifacts and existing orchestration.

The new instruction supersedes the GPU-specific limits of ADR0017 and earlier cloud campaign allocations; their CPU/RAM, evidence and independent-weight requirements remain. The previous archive and review authorized no experiments at the time; this ADR records the subsequent explicit resumption.
