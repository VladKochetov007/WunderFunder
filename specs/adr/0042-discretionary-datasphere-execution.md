# ADR0042: Discretionary DataSphere execution within existing balance

Status: Accepted

Date: 2026-10-18

## Context and evidence

The ranked25 campaign had prepared and reviewed the first A100 profile but was waiting for a new numeric allocation. The user now explicitly instructs: "сколько хочешь на датасфере делай". This is current spending authorization; the historical24A100-hour campaign is not its basis. The [authority update](../../research/campaign_2026-10-18/priority25_execution_r4/receipts/DATASPHERE_AUTHORITY_UPDATE_20261018.json) preserves the exact instruction.

## Decision

Root allocates useful DataSphere experiments without another numeric-budget question. The [campaign authority](../../research/campaign_2026-10-18/priority25_execution_r4/AUTHORITY.json) records explicit discretionary authorization; a null numeric ceiling means no user-imposed number, not a launch prohibition. Use the existing balance and current provider quotas. Do not top up or change billing.

Each attempt still has a bounded runtime, durable cancellation, exact source/data/cell identity, cost reconciliation and owned artifact collection. The first prepared g2.1 profile retains a one-hour outer bound and one-job concurrency. Subsequent allocation follows measured throughput and actual resources; this initial profile bound is not a new overall campaign ceiling. Beleriand still uses one selected GPU, and its F0/seed1001 pair must not be duplicated. Both arms of any paired cell run on the same provider/runtime.

## Alternatives and consequences

Continuing to wait for a numeric answer would contradict the latest user authorization. Starting unmonitored cloud jobs would leave avoidable cost and collection uncertainty. Explicit discretionary allocation with bounded jobs removes the obsolete wait while preserving measurable execution. No model-quality or provider-admission result is implied by this decision.

## Revisit when

The user changes spending scope or stops; the available balance, quota, throughput, reliability or scientific evidence changes enough to affect allocation.
