---
kind: method
status: inconclusive
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: []
experiment_ids: []
evidence_paths: [submissions/10-14-cpu-policy-r1/README.md]
---

# EPYC9224 bandwidth envelope and CPU probe interpretation

## Question or claim

How much inference headroom might remain in the original four-member endpoint? This is a derived comparison, not a measurement of cache traffic or a certified maximum.

## Mechanism and method

The [shape accounting](../../campaign_2026-10-14/cpu_student_frontier_r1/THEORY_AND_DISCRIMINATING_TESTS_R1.md) gives 4,019,712 matrix coefficients in the actual original ensemble, including its wider third-member head. At one logical byte per coefficient, 39.4 million rows and the reported 3,016-second completion imply 76.55 microseconds/row and 52.51 decimal GB/s of logical weight use.

[AMD's EPYC9224 specification](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9224.html), rechecked October15, lists 2.5GHz base, up to3.7GHz boost,64MB L3 and460.8GB/s socket memory bandwidth. The socket bandwidth is not an allocation to one vCPU. [AMD ZenDNN's architecture table](https://github.com/amd/ZenDNN/blob/main/benchdnn/doc/perf_counters.md#architecture-support), retrieved through its raw source October15, uses32B/cycle L2 fill for Zen4 and64B/cycle for Zen5. Only that architecture constant is used here; its performance-counter aggregation recipe is not adopted.

Multiplying32B/cycle by2.5–3.7GHz gives80–118.4GB/s. Logical weight use divided by that range is44.4–65.6%. Under the deliberately optimistic assumption that every logical weight byte traverses this link once per row at peak rate, weights alone require22.3–33.0 minutes. Relative to50m16s, that corresponds to an idealized1.52–2.25x envelope before other costs.

## Evidence and interpretation

The endpoint already achieves substantial logical throughput; the460.8GB/s socket specification does not imply a nearly9x speedup. Scheduling, vector width and fusion can still matter, motivating the [same-weight A/B/C/D submission probes](../../../submissions/10-14-cpu-policy-r1/README.md). None has a new competition-host result yet.

## Limits and validity

Logical coefficients are not measured bus bytes. Padding, mixed matrix precision, reuse, metadata, cache residency, prefetching and other traffic alter the relationship. One vCPU may share a physical core; actual frequency, contention and quota timing are unknown. Quantization, nonlinearities, recurrence dependencies, preprocessing, Python/native calls and scorer overhead remain. This is neither an achievable runtime forecast nor proof that cache bandwidth is the sole bottleneck. Current compressed W4 expands to byte kernels; genuinely packed W4 would change this accounting and must earn its benefit through measured runtime and quality.

## Decision and next step

Keep the original endpoint as repeated control. Interpret a single faster candidate as provisional, using A–B–C–D–A and later reversed candidate order when useful. Do not infer CPU capacity from ZIP compression, instruction width or local Intel timings alone. Continue student-quality experiments while uploads are pending.

## Source records

- [Fresh ZIP identity/CRC receipt](../../../submissions/10-14-cpu-policy-r1/HANDOFF_FILE_VERIFICATION_20261015_R1.json).
- [Independent implementation review](../reviews/CPU_POLICY_IMPLEMENTATION_REVIEW_20261014_R1.md), covering complete-sequence parity and asserted implementation selection.
