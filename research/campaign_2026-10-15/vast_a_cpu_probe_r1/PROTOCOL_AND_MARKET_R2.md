---
kind: method
status: proposed
created_utc: 2026-10-15T10:35:00Z
updated_utc: 2026-10-15T10:42:00Z
hypothesis_ids: [H-CPU-POLICY-20261014, H-STUDENT-LADDER-20261014]
experiment_ids: [E-VAST-A-CPU-FEASIBILITY-20261015-R1]
evidence_paths: [USER_AUTHORITY_R2.json, EXACT_CPU_SEARCH_R2.json, EXACT_CPU_EXPANDED_SEARCH_R3.json, CPU_ONLY_SEARCH_R4.json, CPU_COMPUTE_SEARCH_R5.json]
---

# Exact CPU access and informative short runtime experiments

## Question or claim

The user approved up to$3 on Vast-A and requires the same CPU as the competition. The current official guide explicitly identifies AMD EPYC9224, family25/model17/stepping1, one24-core socket/48logicalCPUs, 1MiBL2/core and64MiBL3 across4instances. The one-vCPU quota is separate from visible host CPUs. The earlier9354P proposal is therefore not the requested match. The latest message says no GPU is needed; no GPU bundle is being rented while this is clarified.

## Mechanism and method

Prepare the complete packet before any rental: minimal Python3.11/NumPy, pinned native libraries and model weights, no remote compilation/training/full-parquet upload. Freeze eight complete20k sequences from distinct fitting components using a declared seed. Upload112rawfloat32features and need_prediction only, about72MB, no targets/teachers/scoring masks. The existing xteam_h4 ensemble, coefficients, normalization and quantized representation stay fixed across A/B/C/D. Verify CPU model, exposed ISA, selected dispatch and artifact hashes before timing; a matching SKU does not guarantee the same frequency/cache contention/NUMA behavior.

Use one pinned logicalCPU, single-thread libraries and, if exposed writable delegation permits, an explicit oneCPUquota and16GBRAMcap. Report actual controls: affinity is not proof of an exclusive physical core, and lack of cgroup delegation must not be called an enforced quota. First-touch input and model on thatCPU/node. Do not alter host governors, SMT or other tenants. Record topology, cpu.max, cpu.stat, processCPUtime and walltime. A second allowed CPU can provide a separate robustness panel if time permits; it is not an independent host replication.

Use balanced randomized A/B versus B/A comparisons in short blocks, interleave A controls, and process every row with ordinary reset and warmup semantics. Preserve at least six paired blocks per candidate where the pilot shows that fits within budget. Separate cold import/construction from warm callbacks, keep validation/hash work outside the timed callback interval, and do not reorder rows or shorten causal histories. Pair the same sequence and block across policies. A full8sequence panel is160krows, roughly0.4% of official39.4mrows; repeated panels are much more informative than one long unpaired pass.

Report repeat variation conditional on each sequence separately from variation across sequences. Summarize paired log-runtime ratios and block-level uncertainty, medians and tail percentiles, clock/throttling/drift diagnostics, plus absolute microseconds/row. Do not count millions of rows as independent timing replications. After a minimum sixblocks, an exploratory adaptive rule can add blocks up to12 for uncertain comparisons, targeting roughly1% relative interval halfwidth. Optional stopping makes this exploratory precision evidence: reserve a final fixed confirmation panel for the chosen policy and report every attempted variant. If drift/noise remains large, report the unresolved ranking rather than force a winner.

After engine comparisons, spend remaining time on the eight-shape callback ladder and scalar/AVX2 minGRU pair; label synthetic-weight cost separately from trained quality. Use tiny matrix/cell probes to explain an end-to-end result, not to certify capacity. If a contender is within a small fraction of the60minute deadline, retain a full endpoint/official transfer check. A sampled known endpoint may calibrate scale, but its50m16s historical portal duration is not a universal multiplier.

## Evidence and interpretation

Authenticated exact9224 searches returned no verified on-demand compute offers. Removing verification found machine153365, oneH200NVL bundle at$3.6037037/hour with20GBdisk (about$2.10/35min), or interruptible floor$2.6703707/hour; only one distinct host, so not a replication. CPU-only search num_gpus=0 found five exact9224 listings, all resource_type=disk. A resource_type=cpu request failsHTTP400 with the provider enum network_disk,disk,gpu. These are storage listings, not established executable CPU offers; do not create a volume to learn whether it can run code. Vast documentation describes GPU compute and separate volumes APIs. No rental or measured EPYC result exists.

Other-provider search found Contabo's published exact9224 dedicated configuration, monthly pricing far above$3. Generic VPS pricing does not guarantee9224placement/ISA or short billing. Search snippets or old VPS benchmarks are not an immediately purchasable hourly exactCPU offer. No third-party purchase/account change is authorized by the Vast-A instruction.

## Limits and validity

A different Zen4 SKU is not the user's requestedCPU. Dualsocket9224 still needs local affinity/first-touch controls and has different shared-host conditions from the official singlesocket sample. Runtime-only fitting fixtures establish neither prediction quality nor unseen distribution coverage. Failed market/API requests are execution evidence, not evidence against kernel/architecture mechanisms.

## Decision and next step

Keep rental spend atzero until a launchable CPU-only exact9224 route is identified or the user permits using the CPU of a GPU bundle. Packet/fixture preparation continues within local bounds and remains useful for competition-host probes. Existing local training/research authorization continues independently.

## Source records

- [User authority](USER_AUTHORITY_R2.json), [verified exact search](EXACT_CPU_SEARCH_R2.json), [broader exact search](EXACT_CPU_EXPANDED_SEARCH_R3.json), [zeroGPU search](CPU_ONLY_SEARCH_R4.json), [resource-type diagnosis](CPU_COMPUTE_SEARCH_R5.json).
- [Official competition CPU specification](https://wundernn.io/connectome/docs/submission_guide).
- [Vast compute search API](https://docs.vast.ai/api-reference/search/search-offers), [storage volumes guide](https://docs.vast.ai/guides/instances/storage/volumes).
- [Contabo published pricing](https://contabo.com/en-us/pricing/).
- [Minimal official Python image pin](IMAGE_PIN_R1.json) for a future admitted runnable packet; no image pull/rental.
