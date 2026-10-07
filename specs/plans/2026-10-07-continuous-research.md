# Continuous research across local, Kaggle and Vast compute

Status: user approved implementation on 2026-10-07. This preserves the agreed plan; execution evidence belongs in the campaign ledger. A prepared packet or an inaccessible observer does not establish a running experiment.

## Subsequent online-learning allocation

The user subsequently added online learning as a new long-term family, requesting independent Grok, Freebuff/GLM, OpenCode/Muse and Kimi ideas. [E530 synthesis](../../research/campaign_2026-10-06/bandit_continuation/online_learning_E530/SYNTHESIS_V1.md) registers eight arms/interactions with null numerical priors, including cheap causal feature dynamics and learned fast-weight teachers. Use the existing protected exploration allocation; this does not block the leading distillation/teacher fits or expand the paid budget. Infer only from currently observable inputs; all adaptive state resets each sequence and test targets are unavailable. Larger research teachers retain a compression path. Requested-provider local access failures are recorded separately from completed native proposals and CPU protocol checks.

## Binding ownership addendum

The user subsequently instructed: **“touch only machines you rented. do not hurt existing, they are not mine.”**

Vast mutations, SSH, collection and destruction may address only instances created by this campaign, with a persisted creation receipt binding the unique attempt, campaign label and provider instance ID. Existing machines are outside scope. A matching label alone is not ownership. Never bulk-stop, bulk-delete, adopt a pre-existing instance, change its configuration, access its filesystem or interrupt its jobs. Reconcile ambiguous creation without issuing another create or touching an unproven instance. Keep unrelated instance identifiers out of public research artifacts.

## Objective and allocation

### Subsequent offer-specific selection

**Latest steering supersedes the shortlist below:** the user explicitly said to choose independently across the market, because their examples are not necessarily best. No offer is selected. Search all suitable current offers and compare whole-attempt cost and measured throughput; the two examples below remain historical reference points only. Do not hardcode either ID into search or admission logic.

The user supplied RTX 3090 offer **54036773** (India, unverified, 24 GB VRAM, four i5-6402P CPU cores, 24 GB host RAM, 284 GB available disk, quoted $0.108/hour plus bandwidth) and authorized efficient discretionary selection. Prefer this offer for a bounded first pilot if it is still available and actual storage/transfer plus useful execution fit the $4 ceiling. This is a scoped exception to the initial preference for verified hosts with 32 GB RAM, not a waiver of measured resource headroom, ownership or billing gates. Limit host working memory to at most 80% of its actual allocation and use one worker. The quoted rate and marketplace reliability are user-supplied observations, not verified live pricing or a measured probability of job completion. Six compute hours at that quote are $0.648 before other charges.

Do not select the supplied three-A2000 offer 16670765 for the initial full-data cohort: its listed 18 GB disk cannot hold the approximately 35 GB supplied data, each GPU has its own 6 GB VRAM, and all three share four CPU cores and 16 GB host RAM. A different quote with sufficient resources could be evaluated later. Nothing has been rented at this selection boundary.

Keep each available compute lane supplied with useful, bounded experiments, preparing its successor before the current recipe finishes. Allocate approximately 55% to promising existing directions, 30% to independent exploration, and 15% to confirmation, serving and necessary optimization. Reflect every two hours and after meaningful results. Numerical priors stay null until relevant measurements exist; confidence intervals are not automatic arm-kill gates.

The private `.env` entry `vastai` supplies the Vast credential. Never log, commit, include it in an LLM prompt, or transfer it to a rented worker. User-approved spend is at most **$4 for the first evidence-bearing cohort and $20 cumulatively**, including compute, storage, transfers and cleanup. Release the remaining $16 only from completed evidence; no top-up. This supersedes the earlier blanket prohibition on paid rentals only within this allocation.

## Level 1 scientific cohort

| Direction | First comparison | Placement and rationale |
|---|---|---|
| H230 heterogeneous distillation | Frozen GRU256×2, own audited combo initializer, eight-source raw target, distillation weight 0.7, family-head weight 0.1, B16/chunk2000, FP16, two epochs, fixed epoch 1. Compare with a matched family-head-off control; seed 3 first, then seed 4. Existing C1 is contextual evidence, not the isolated family-head ablation. | Local GPU first. Source cache and full-shape resource profile exist; an actual student result is missing. |
| H232 wide teacher × event features | GRU384×2 with live causal event features versus the existing same-seed raw E521 teacher. Preserve four epochs, fixed epoch 3, split, optimizer and target treatment; seeds 3 then 4. Reuse valid raw outputs. | Kaggle GPU. One declared input-view change with independently trained controls. |
| H217 independent causal TCN | Scratch C128/L12/mix1, raw inputs, seed 7, four epochs, fixed epoch 3. Profile B8 first. Compare against raw GRU256×2 seed 7 with the same training schedule, reporting actual unequal parameter counts and costs. Test fixed-student/blend contribution even if standalone score is weaker. | First Vast cohort, or Kaggle if Vast access/staging prevents a useful run. Repair missing finite/AMP-update profile evidence before fitting. |
| H233 conditional readout | Freeze own C1 seed-3 GRU256; rank-32 hidden/context projections; zero-initialized two-target residual. Context is the existing 99 causal event features in frozen order. Compare unchanged anchor, hidden-only, additive hidden/context and multiplicative interaction under identical fitting membership and loss. Stream states. | Next available GPU, bounded CPU grading. Low-cost route to a useful submission and a contextual-state interpretation test. |

Protected architecture preparation proceeds in parallel:

- TPU: E463 V4 unequal-weight gradient/reduction diagnostic, then bounded real-data Torch/JAX parity. After parity, projection hoisting and independent models per chip, followed by H216 associative linear-recurrence teacher.
- H218: independent block-recurrent causal attention prototype and complete-update profile; its first scientific fit receives an exploration slot in the next cohort. A negative TCN result does not close attention.

All fits retain official evaluation clipping, declared fitting ancestry and existing exposure labels. Local deltas are not public forecasts.

## Level 2 evidence branches

- If H230 helps, separate raw/clipped family-target and coefficient-normalization tests; confirm and package the useful student.
- If H230 disappoints, use the family-head ablation and family residuals for a bounded source-removal/reweighting test, not another unchanged continuation.
- If H232 helps or adds complementary signal, complete H239 compact/wide × raw/event-feature factorial and same-fixed-student comparisons. A wide-only feature gain does not establish a scaling interaction.
- If TCN/attention adds conditional value, register recurrent-plus-new-family interaction nodes and test fixed blending and distillation. Standalone rank is not the admission criterion.
- If H233 helps, test H237 shared-trunk readout ensembles, then bounded trunk unfreezing.
- If residuals suggest missing slow context, test H235 fixed/learned multiscale memory and H238 raw-context/normalized-innovation representations. First establish H231's concrete distinction from existing two-clock designs.

## Compute and dispatch

Local aggregate: 6 CPU cores, 20 decimal GB RAM, 5 decimal GB observed VRAM, one GPU worker. Target aggregate resident usage below 16 GB, memory-high 18 GB and no swap. H230 receives a resource-only successor with a 9 GB worker ceiling, 3 CPUs and unchanged scientific recipe; record anonymous/file memory separately. Support jobs share the remaining allocation.

Kaggle: actual device capacity with at least 20% RAM/VRAM headroom and fresh quota admission. Use E520's explicit-offline V11 successor after transport review; prepare a subsequent fit packet concurrently. Two visible GPUs may run independent fits only after measuring shared host/input headroom. Replace the historical TPU probe-only cap with bounded production allocations after parity.

Vast: start with one verified on-demand GPU with at least 24 GB VRAM, 32 GB host RAM and sufficient measured CPU/disk capacity. Select among current eligible offers by estimated whole-attempt cost, then update from actual throughput. Freeze the useful recipe before rental. No rental if staging leaves insufficient useful run time within the $4 initial ceiling. Scale out only when completed experiments per dollar justify it. Collect and hash artifacts, destroy only campaign-owned instances when their useful queue empties, and verify teardown. Stopped instances still incur storage charges (Vast official pricing documentation: https://docs.vast.ai/guides/instances/pricing).

Extend existing finite queues with a small provider ledger and Vast adapter. Bind unique attempt, source/input hashes, resource/cost reservations, predecessor, provider identity, phase heartbeat, expected artifacts and cleanup. Persist intent before submit; reconcile ambiguous submission before retrying. Provider terminal status without verified artifacts is incomplete. Keep one reviewed successor per active lane; no blind retries, unbounded background loops or duplicate attempts.

## Diverse agents

Native delivery/operator/critic roles use gpt-6-luna with max reasoning, at most three subordinate roles. Grok 4.7 supplies independent architecture proposals; OpenCode supplies a separate implementation/critique; Freebuff supplies a random-literature transfer exercise. Use actual verified models, one active session per credential profile, independent initial prompts, directed and seeded random passages, several distinct components including an unconventional one, and public outputs only. Ideation availability does not gate runnable science.

Diagnose actual environment failures rather than declaring providers unavailable. Grok session creation currently fails on read-only state. OpenCode startup aborted under a restrictive address-space limit; use actual cgroup memory limits when an authorized host execution path is available. Freebuff's interactive route remains unverified. Do not bypass managed restrictions or expose credentials to repair access.

## Acceptance and evidence

- Verify source/data membership, causal-prefix/reset behavior, finite gradients, complete-update memory, attempted/completed/AMP-skipped updates before a fit.
- Test dispatcher restart, ambiguous submission, empty queue, missing outputs, hash mismatch, quota/cost exhaustion and verified teardown. Test that all unrelated/pre-existing instance mutations are rejected before any HTTP/SSH action.
- Repair E528's nested receipt lookup under a new attempt ID, preserving V1. Require exact decoded-feature parity and actual amortization before adopting caching. Do not build the roughly 107 GB full fitting-feature cache locally.
- Completed fits deliver checkpoints, predictions, official-metric grades, measured cost and tree updates. Independent review precedes promotion.
- Preserve failed/superseded artifacts. Update notes, frontier and tree/index; commit implementation, evidence and packaging as meaningful separate steps when Git writes are available. Keep `.env` ignored and use explicit staging paths.
