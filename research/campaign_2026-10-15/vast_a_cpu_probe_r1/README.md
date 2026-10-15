---
kind: experiment
status: proposed
created_utc: 2026-10-15T08:42:36Z
updated_utc: 2026-10-15T08:42:36Z
hypothesis_ids: []
experiment_ids: []
evidence_paths: [OFFERS_R1.json]
---

# Vast-A Zen 4 CPU benchmark proposal

## Question or claim

Can a short Genoa rental resolve relative native-kernel and architecture costs faster than asynchronous competition uploads? The latest user names Vast-A (the `vastai` environment credential, never `vastai2`) and explicitly requires approval before rental. Read-only search is authorized; no rental or billing change is authorized yet.

## Mechanism and method

Proposed exact offer 50293333, machine 140925: AMD EPYC 9354P 32-Core Emb Processor, Oklahoma, one bundled RTX6000Ada, 20GB disk. API quote $0.6718518519/hour including disk, about $1.344 for two hours excluding traffic. Request a maximum two-hour session and $2 total budget from existing prepaid credit ($5.1566 snapshot); no top-up, no automatic extension or replacement rental. Verify price and availability again before creation. The bundled GPU is not needed for this CPU experiment.

After approval, prepare the immutable benchmark packet before creating the instance. Use Python3.11 and the existing original-weight A/B/C/D libraries and model artifacts, pin all hashes. Validate actual CPU/ISA, CPU quota, affinity, SMT topology, cgroup membership, memory and library dispatch before timing. Time one worker sequentially under one CPU quota and 16GB maximum RAM, with pinned logical CPU. Run matched complete20k callbacks with resets, warmups and repeated controls in A/B/C/D/A and reverse order; check whole-sequence predictions and final state. Then run the existing eight-shape cost ladder and minGRU scalar/AVX2 pair. Collect timing distribution, CPU time, throttling, memory, binary and artifact hashes. If feasible within the same session, run longer original endpoint replays to quantify drift. No model fitting or architecture-quality claim.

Collect evidence and tear down this exact owned instance before the two-hour deadline or earlier when complete. Local watchdog and instance-local timeout must be configured before useful work; retain cleanup receipts. Spending ceiling is a controller budget, not a provider-guaranteed billing cap. Abort and promptly clean up if the requested ISA/quota is unavailable. No new rental or fallback without the user's approved scope.

## Evidence and interpretation

[Sanitized live offer receipt](OFFERS_R1.json) records the authenticated Vast-A search and exact 20GB quote. No credentials or provider IPs are archived. Cheaper inspected EPYC offers were older generations. A 9684X offer has extra cache and is less suitable as the first comparator. SDK discovery is operational evidence only; no rented-host result exists.

## Limits and validity

This is a Zen4 proxy, not an exact competition-machine match. Cache topology, frequency, SMT neighbors and shared-host load can affect absolute latency and even rankings. Calibrate with the existing endpoint whose observed competition duration is50m16s, without assuming a universal scaling factor. Retain final competition submissions to validate selected kernels and deadline headroom. Existing callback fixtures are exposed fitting data; they support runtime/correctness rather than held-out prediction-quality claims.

## Decision and next step

Await explicit user approval of offer50293333 for at most two hours and $2 total. No instance has been created. Continue healthy local fits independently.

## Source records

- [Complete CPU/student plan](../../campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md).
- [Existing A/B/C/D checked packages](../../../submissions/10-14-cpu-policy-r1/README.md).
- [Vast offer API](https://docs.vast.ai/api-reference/search/search-offers).
- [AMD9354P specification](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9354p.html).

## October15 successor: exact9224 requirement and $3 authority

The original9354P proposal above is superseded by the latest user requirement for the same CPU as competition. [Revised protocol and market findings](PROTOCOL_AND_MARKET_R2.md) records approved$3Vast-A scope, exactEPYC9224officialspecification, randomized complete-sequence timing design, and the unresolved CPU-only access route. No rental has occurred. Zero-GPU offers are storage, not measured CPU compute. [Authority](USER_AUTHORITY_R2.json) and all linked search receipts preserve the actual attempts.
