# ADR0014: Continuous useful compute until explicit user stop

Status: accepted, 2026-10-08 Europe/Kyiv.

## Context

The user explicitly requires sustained use of Kaggle GPUs/TPUs, local compute and authorized Vast resources until they say to stop. Previous campaign episodes produced prepared recipes and monitoring infrastructure without actual fitting. Full host/network access is restored, so historical sandbox failures do not justify continued inactivity. The long-term objective needs timely empirical results across independent arms.

## Decision

Use a rolling execution queue with an accountable operator for each admitted compute lane. Operators own input/profile checks, actual launch, phase monitoring, diagnosis and identified retries, checkpoint/prediction collection and grading handoff. Code and packet preparation are intermediate milestones. Start a valid runnable arm promptly; unrelated implementation, ideation or ablations must not gate it.

Prepare a reviewed successor during current execution. At terminal status, collect and verify actual artifacts, reconcile quota/cost and immediately admit the next useful ready attempt after fresh resource checks. Keep local GPU execution serial under its shared lock; cloud concurrency is determined by measured device and host headroom. Use live Kaggle quota with its required reserve, and replace historical TPU probe budgets with bounded production allocation once actual-device numerical and real-data parity pass.

If a lane is idle, write the actual reason, repair owner and concrete next action. A prepared queue, provider status, proposal or CLI access check is not an active experiment. Controller updates must say which stage is actually running. Every two hours, reassess allocation from measured effects, cost, conditional contribution and information value while retaining exploration across families.

This instruction supersedes historical local time cutoffs and stop snapshots. The latest explicit stop closes new admission and drains healthy jobs; a later resume is recorded separately. Do not automatically stop the campaign because one lineage plateaus or an uncertainty interval crosses zero.

## Binding boundaries

Local aggregate remains 6 CPU cores, 20 decimal GB RAM and 5 decimal GB observed VRAM, one GPU worker, with 18 GB memory-high and no swap. Cloud resources have separate measured headroom. Do not cause avoidable OOM or raise caps to bypass admission. All fitting ancestry, independent implementation/weights, causal serving and official evaluation constraints remain binding.

Vast stays within $4 initial cohort and $20 cumulative all-in, with later release based on completed evidence. No subscription or top-up is authorized. Only campaign-created instances with durable creation receipts may be accessed, modified, collected or destroyed. Never adopt or touch pre-existing machines. Paid workers must have bounded monitoring/collection/teardown, and must be destroyed once their useful queue is empty.

Do not duplicate completed experiments, run no-op workloads, fabricate utilization or rent before a useful recipe and bounded lifecycle exist. Empty queues create preparation/repair work, not authorization for idle spend. Public upload retains the human gate.

## Consequences and links

This is an execution requirement, not a numerical promise or a replacement for scientific allocation. Leading H230 student delivery proceeds independently of cloud preparation; exploratory architectures and online learning receive their protected allocation. Actual timing replaces planning ranges as results land.

- [ADR0010: experiment delivery](0010-continuous-experiment-delivery.md)
- [ADR0013: bounded owned paid compute](0013-owned-paid-compute-and-continuous-queue.md)
- [Approved execution plan](../plans/2026-10-07-continuous-research.md)
- [Live estimates and owners](../plans/2026-10-07-live-research-estimates.md)
