---
kind: review
status: complete
created_utc: 2026-10-15T11:00:00Z
updated_utc: 2026-10-15T11:00:00Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - ../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_ONLY_SEARCH_R4.json
  - ../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_COMPUTE_SEARCH_R5.json
  - ../../campaign_2026-10-15/vast_a_cpu_probe_r1/EXACT_CPU_EXPANDED_SEARCH_R3.json
---

# Vast-A zero-GPU offer availability review

## Question or claim

Do the current Vast-A searches expose a launchable CPU-only compute offer for the exact AMD EPYC 9224 required for a useful competition-runtime comparison, under the user's latest scope that does not need a GPU?

## Method and evidence

This is a bounded source and receipt review. I made no provider API call, created no instance and did not attempt an offer. Root's read-only R4 search asked for `num_gpus == 0` and the exact EPYC 9224 CPU name; R5 tested the `resource_type=cpu` filter. R3 retains the prior expanded exact-CPU search, which found GPU-bundled results.

The R4 exact-CPU query returned five rows. All five are labeled `resource_type: disk` and `gpu_name: N/A`, despite reporting the CPU name and an effective-vCPU count. The generic zero-GPU search likewise returned disk-typed rows. R5's `resource_type=cpu` query returned HTTP 400 and the API error lists accepted resource types as `network_disk`, `disk`, `gpu`, or null. An earlier `resource_type neq disk` query also failed. These are not evidence that a disk offer supplies a CPU container.

Vast's [official search-offers API reference](https://docs.vast.ai/api-reference/search/search-offers) describes `/bundles` as a search for GPU machine offers; its documented filters and response examples are GPU-oriented, and the example response has `resource_type: gpu`. The [REST API overview](https://vast.ai/developers/api) describes the marketplace resources as GPU cloud resources. The official API also exposes separate [volume search](https://docs.vast.ai/api-reference/volumes/search-volumes) and [volume rental](https://docs.vast.ai/api-reference/volumes/rent-volume) endpoints. Together with the observed `resource_type=disk` results, this supports classifying R4 as storage/disk listings rather than established CPU compute. Public documentation does not prove that Vast has no private or newly added CPU-only endpoint; it simply provides no documented CPU-compute offer path here.

The earlier exact-CPU R3 expanded result was an H200-bundled machine: its one-GPU on-demand offer was about $3.604/hour and marked unverified; the two-GPU offer cost more. It is outside the latest no-GPU scope and cannot be admitted by the current campaign controller. `scripts/vast_campaign.py` explicitly requires verified on-demand offers, exactly one GPU and at least 24 GiB GPU RAM; do not weaken this GPU workflow to treat a disk row as compute.

## Interpretation and limits

**Current disposition: hold Vast execution.** The available receipts establish no CPU-only compute offer for the exact competition CPU. They also do not justify concluding that all Vast CPU-only compute is impossible: the endpoint error and public docs bound what this search route documents, not every provider product or private API. Root can continue read-only confirmation, but no H200 bundle or different CPU should be substituted under the user's latest instruction.

If a real exact-CPU compute offer is found, its identity and billing resource must be established before admission: fresh on-demand quote; verified/rentable status and instance-create semantics; exact CPU model; actual socket/core/SMT/NUMA/cache/ISA exposure; one-core quota and affinity; actual 16 GB memory limit; all-in creation-to-destruction cost including disk and transfer within $3. API `cpu_cores_effective` is not evidence of the guest's usable topology. Exact CPU model alone does not control frequency, microcode, host contention, cache sharing or virtualization.

For any later runtime comparison, use the unchanged original A/B/C/D callback packages and weights, one CPU thread with pinned affinity and fixed library thread counts, full 20,000-row sequences with state reset at sequence boundaries, and randomized/interleaved repeated candidates plus early/late A controls. Record wall and CPU time, frequency, cgroup throttling, memory, source/build/package hashes and callback outputs. Label a remote result as a same-CPU proxy unless relevant competition hardware topology and runtime conditions are matched; a single warmed-sequence latency is not whole-test runtime. This is runtime evidence only, not model-quality evidence.

## Decision and next step

No rental is admitted by this review. Continue only the root-owned read-only exact-CPU availability check. If no bona fide CPU compute offer exists within scope, close this Vast path without adapting the GPU-only controller or choosing a different processor. Preserve the R3/R4/R5 search receipts as access/product evidence, not scientific results.

## Source records

- [R3 exact-CPU expanded search](../../campaign_2026-10-15/vast_a_cpu_probe_r1/EXACT_CPU_EXPANDED_SEARCH_R3.json)
- [R4 zero-GPU search](../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_ONLY_SEARCH_R4.json)
- [R5 CPU resource-type query](../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_COMPUTE_SEARCH_R5.json)
- [Vast campaign offer admission logic](../../../scripts/vast_campaign.py)
- Official sources: [Vast search-offers API](https://docs.vast.ai/api-reference/search/search-offers), [Vast REST API overview](https://vast.ai/developers/api), [search volumes](https://docs.vast.ai/api-reference/volumes/search-volumes), and [rent volume](https://docs.vast.ai/api-reference/volumes/rent-volume).
