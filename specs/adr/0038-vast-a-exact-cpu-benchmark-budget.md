# ADR0038: Vast-A exact-CPU benchmark budget

Status: Accepted budget and constraints; rental access unresolved.
Date:2026-10-15.

## Context and evidence

The user approved up to$3 on Vast-A, explicitly naming credential `vastai` rather than `vastai2`, asked for efficient repeated subsample timing and the same CPU as competition, then clarified that a GPU is not needed. The current official guide identifies AMD EPYC9224. The older9354Pproposal is superseded; it was not an exactCPU match.

## Decision

Root may spend at most$3total on the CPU benchmark within these constraints, with no top-up or billing change. Prepare the immutable experiment/fixtures first; minimize paid setup and repeated full-data runs. Preserve complete causal sequences, fixed same-weight native comparisons, randomized paired blocks and independent timing-noise/sequence-variation summaries. Require actualCPU/ISA/dispatch verification, durable deadline/owned-instance cleanup and bounded artifacts before launch. Keep other authorized research running.

Read-only Vast-A searches establish exact9224 only in GPUbundledcompute; zeroGPUmatches are diskvolumes and resource_type=cpu is rejected. No rental has occurred. Do not rent storage as speculativecompute or substitute a differentCPU. Root asked whether using only the bundleCPU is acceptable given the latest user clarification; that dependent action waits for an answer.

## Alternatives and consequences

[Protocol, live receipts and authority](../../research/campaign_2026-10-15/vast_a_cpu_probe_r1/PROTOCOL_AND_MARKET_R2.md) preserve all attempts and provider limitations. The earlier no-new-paid-capacity rule is superseded only for this explicit$3CPU experiment; other paidGPUcapacity remains unauthorized. SameSKU still needs affinity/NUMA/host-load controls and final official runtime confirmation.

## Revisit when

A documented CPU-only exact9224 offer appears, or the user clarifies that CPUuse of a GPUbundle is acceptable. New spending outside$3 or a differentCPU requires changed user scope.

## Clarification received

The user selected **Keep looking for CPU-only access**. GPUbundle rental is excluded. The$3budget and exactEPYC9224requirement remain; no paidaction has occurred. [Clarification receipt](../../research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_ONLY_AUTHORITY_R3.json).
