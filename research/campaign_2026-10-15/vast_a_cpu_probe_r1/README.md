---
kind: experiment
status: prepared
created_utc: 2026-10-15T08:42:36Z
updated_utc: 2026-10-15T10:44:03Z
hypothesis_ids: []
experiment_ids: []
evidence_paths: [OFFERS_R1.json, CPU_TIMING_FIXTURE_R1.json, PREPARATION_RECEIPT_R1.json]
---

# Vast-A Zen 4 CPU benchmark proposal

## Scope update, 2026-10-15

The later user instruction approved up to$3 for the exact-EPYC9224 Vast-A runtime probe. The older9354P/$2 offer and pending-approval text below are historical; the current fixture task creates no instance and performs no provider action.

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

The original offer50293333/$2 proposal above is historical. A later user instruction approved up to$3 for the exact-EPYC9224 Vast-A work; this fixture-preparation task itself performs no rental/provider action. No instance has been created as part of this fixture task, and the controller owns any later rental execution.

## Source records

- [Complete CPU/student plan](../../campaign_2026-10-14/cpu_student_frontier_r1/PLAN.md).
- [Existing A/B/C/D checked packages](../../../submissions/10-14-cpu-policy-r1/README.md).
- [Vast offer API](https://docs.vast.ai/api-reference/search/search-offers).
- [AMD9354P specification](https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9354p.html).

## October15 successor: exact9224 requirement and $3 authority

The original9354P proposal above is superseded by the latest user requirement for the same CPU as competition. [Revised protocol and market findings](PROTOCOL_AND_MARKET_R2.md) records approved$3Vast-A scope, exactEPYC9224officialspecification, randomized complete-sequence timing design, and the unresolved CPU-only access route. No rental has occurred. Zero-GPU offers are storage, not measured CPU compute. [Authority](USER_AUTHORITY_R2.json) and all linked search receipts preserve the actual attempts.

## Explicit CPU-only clarification

The user answered **Keep looking for CPU-only access**. [CPU-only authority R3](CPU_ONLY_AUTHORITY_R3.json) excludes the GPUbundle and preserves the$3limit/exact9224requirement; spend remainszero. [Other-provider search R6](OTHER_PROVIDER_SEARCH_R6.json) records public CPU-only alternatives: Contabo/BusyFirm exactCPUmonthlycommitments exceedbudget; TensorDock advertises hourlyCPUcloud but exact9224stock is unverified and its advertised initialdeposit starts$5. No third-party purchase/signup/topup. The preparedfixture/harness stays useful while a valid route is absent.

The bounded read-only [public host scout R7](OTHER_PROVIDER_SEARCH_R7.json), checked October15, found no qualified CPU-only exact9224 hourly/per-second offer within$3. Contabo's official page confirms the exact dedicated CPU but lists monthly terms; TensorDock advertises CPU-only pay-as-you-go while public pages do not pin an EPYC9224 host. FireVM and NexSpire exact-CPU listings require GPU bundles and are excluded. This scoped search does not prove global absence; no credentials, signup, API call, purchase, or provider mutation were used.
## Portable CPU timing fixture

[CPU timing fixture manifest](CPU_TIMING_FIXTURE_R1.json) binds eight complete F0 fitting sequences to the frozen split, `train.parquet` footer identity, source-qualified row groups, seed-3 selection order, per-sequence feature hashes, and whole-payload hashes. Its only payloads are little-endian float32 features shaped `[8,20000,112]` and a synthetic runtime request vector shaped `[8,20000]` that skips rows0–98; it contains no targets, official scoring mask, teacher output or cache values. The ignored binary payloads are stored under `artifacts/vast_a_cpu_probe_r1/` on the local physical filesystem, and the committed manifest points to their hashes rather than committing the payloads.

The fixture supports runtime measurement only. The eight sequences are from distinct F0 fitting components; their use establishes no prediction-quality or held-out result. The source-order receipt is the fixture's deterministic seed-3 selection order, not a claim about DataLoader minibatch order. No Vast-A instance was created during fixture preparation.

The preparation completed with exit status0 under `scripts/limited` at one CPU, 1.5GB `MemoryMax`, no swap, and a 180-second runtime cap. The payload directory resolves to the physical Btrfs filesystem. The writer used `SequenceFile.read(..., targets=False, mask=False)`, checked all feature values for finiteness and complete `[20000,112]` shape, and verified each feature slot again from the finished memmap. A second bounded pass recomputed both whole-file SHA-256 values and matched the manifest. The request schedule is synthetic: false for rows0–98 and true for rows99–19999.

The committed [fixture manifest](CPU_TIMING_FIXTURE_R1.json) has SHA-256 `941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05`. It pins feature-array SHA-256 `e8e8e1268cea54be12c3cf93e1be79b18cb097a1584851bcc4f5dc65885dd337`, request-array SHA-256 `976947b85bdde6f53e3e7323e872ab4019175b404e58d29fd40321fadfe8ed82`, and source identity SHA-256 `956724a5f1e695ffdd33d7b7629ec3ae06abf9c647465e2d531574754741d9ad`. The [preparation receipt](PREPARATION_RECEIPT_R1.json) records the exact bounded command, generator/source hashes, binary paths and verification scope. The producer pins the frozen fold-0 split SHA-256 and the Parquet footer, file-size and modification-time identity; it does not rescan the full29GB Parquet body.

Builder source is [prepare_vast_a_cpu_probe_fixture.py](../../../scripts/prepare_vast_a_cpu_probe_fixture.py), SHA-256 `b2717e31431223569eb22041a3f7cd97154ce2c90bc300b9e1250766e55004c2`. The feature reader is the unchanged [SequenceFile implementation](../../../src/wnn/data.py), SHA-256 `a93da840137adbd7afdb79a22687fb73150e77107f91ca2adc5211b854bc7e00`. Binary files remain untracked under the local ignored path `artifacts/vast_a_cpu_probe_r1/` and are not copied into source control.

## Integrated validation snapshot

[Navigationcheckreceipt](NAVIGATION_CHECK_R1.json) records successful fulltreebuild and a strictcheck that passed the earlier graph/ledgercoverage stages but failed generatedcatalog/integrityfreshness while liveH384telemetry was changing. No checkwasrelaxed and no modeljobwasstopped. [Portablepanel and actualPython3.11A/Bsmoke](../../campaign_2026-10-14/cpu_student_frontier_r1/portable_cpu_benchmark_r1/README.md) are integrated; [independenttimingreview](../../unified/reviews/VAST_A_CPU_POLICY_TIMING_REVIEW_20261015_R1.md) clears exploratoryruntimeuseconditionalonproperouterdeadline, not an exactCPUresult or finalwinner.

## Input-only per-field clock screen

The [screen finding](PER_FIELD_CLOCK_INPUT_SCREEN_20261015_R1.md), [measured receipt](PER_FIELD_CLOCK_REDUNDANCY_SCREEN_R1.json) and bounded [runner](per_field_clock_probe_r1.py) reuse only the fixture's hash-verified raw feature bytes. Across eight source-qualified F0 fitting components, 112 per-field ages produced 19–48 distinct reset patterns per sequence versus seven group patterns; group-age and field-age values differed on 7.3%–26.7% of cells. No field was constant, no age reached the cap, and row-position degeneracy was 0.368%. Existing stream/batch and reset parity passed. The input screen found added deterministic representation resolution, not predictive value, and its features-only panel is exposed fitting data. It ran for16.95s on one CPU with a500MB/no-swap limit, peaked at95,277,056 bytes, imported no Torch and recorded zero task-cgroup OOMs.
