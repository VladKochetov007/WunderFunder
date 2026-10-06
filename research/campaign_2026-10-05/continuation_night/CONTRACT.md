# October 5–6 research continuation

Authorization: the user approved the proposed continuation plan and requested
implementation. This contract supersedes the October 5 morning admission cutoff
and 30 GB local RAM limit. Other campaign evidence and budgets remain cumulative.

## Question, deliverable and verification

Which causal representations, supervision, training or deployment mechanisms
improve the complete predictor beyond the confirmed public score 0.6660?
The long-term target is 0.7000, not an achieved or presently supported forecast.
Deliver reproducible experiments, human-readable research and EDA, and eligible
deterministic submission ZIPs. Verification tier C applies to public forecasts;
component-excluded empirical comparisons and independent reproduction are required
before promotion. Primary metric is `global_wp_clip2_v1`; prediction and target
clipping, weights, masks and pooling are frozen in the existing official evaluator.

## Time and budgets

No new local scientific process may start at or after
`2026-10-06T10:00:00+03:00` (`2026-10-06T07:00:00Z`). This includes CPU timing,
probes and retries, and is rechecked immediately before execution. Admitted work
may finish; previously registered scoring, collection and reporting may drain.
Kaggle work may continue within its separately measured free quotas.

The cumulative ceiling remains 80 device-GPU hours; TPU quota is separate.
Initial read-only accounting observed 13.4753465044 charged local GPU hours,
10.1565066667 provider GPU hours, and a conservative total of 33.7883598377
device-GPU hours (provider usage charged at two devices). The existing local001
block has 37,888.752584154 seconds remaining of its 24-hour allowance. A new
controller must carry this spent allowance forward, never create another 24-hour
budget merely by creating a fresh state directory. Refresh quota and attempts
before reservations. No paid provisioning or competition uploads.

Allocate remaining GPU compute 60% to complete-predictor/registered comparisons,
20% to unusual mechanisms and 20% to independent reproduction/recovery. Preserve
four provider TPU hours for recovery. Failures and allocated idle devices count.

## Resource contract

All local research descendants, builds, transfers and notebook work share an
aggregate 12 CPU / 40,000,000,000-byte RAM ceiling, a 36,000,000,000-byte soft
RAM threshold and no worker swap. Reserve two CPUs and four GiB for support.
One local GPU worker; default allocator allowance 4.2 GB and observed-device
stop at 4.8 GB, never exceeding the user's 5 GB ceiling. Smaller reviewed pilot
limits remain smaller. Projected additional allocation must leave eight decimal
GB of system MemAvailable. Never kill unrelated user processes.

Kaggle has separate headroom: target at most 70% of effective host RAM and stop
at 85%; target at most 80% accelerator memory and stop at 90%, subject to any
stricter existing limits. Start one scientific worker per notebook; two workers
require measured full-shape aggregate admission. GPU runs existing PyTorch science
while TPU actual-device correctness and production-shape profiling gate E465.
Maintain at least 20 GB local free disk; stream components and reuse immutable
inputs. Compressed NPZ members are not memory-mapped by `mmap_mode`.

## Experimental invariants

Register every distinct mechanism from all ten `reflect_05_day/*.md` reviews,
including speculative ideas. Historical negatives do not veto changed tests.
E415/E418 already ran; audit their actual results and implement parent-preserving
follow-ups. The ab85 `gclpall` holdout result is fitted; PUB143 is an inferred
mixture scenario. Resource/ABI/provider failures are execution-censored.

Freeze controls, source/data/checkpoint identities, evaluator and contrasts before
fitting. Preserve parent, zero-init and capacity controls and require recovery of
an injected predictable correction before interpreting repaired screens. Keep
production quiet-row updates and causal sequence resets. Origin-privileged future
information is teacher-training-only and never enters the deployed student.
Keep component/ancestor exclusions and explicitly label exposed panels.

Use existing preregistrations; default new full-recipe screens to seed3/fold0
and fixed ep3, with seed4/fold0 and seed3/fold1 confirmation for promotion.
Report complete-predictor and parent effects separately, both targets,
class-by-event-age diagnostics and 2,000 paired component bootstrap draws.
Pure regression controls receive no classification auxiliary by accident.

## Independence, failures and memory

Root owns budget, contract, lineage and arbitration. At most two implementation
actors and an independent critic/grader/reproducer; depth at most two. Actors
cannot approve their own results or change frozen graders, permissions or budgets.
Competing scientific implementations use disjoint directories or worktrees.
No actor launches science before source review and controller admission.

Sample resources and preserve exact commands, seeds, code/input/output hashes,
checkpoints, predictions, logs, exit/signal status, OOM counters, peaks and costs.
Check heartbeat/status every five minutes and provider quota every fifteen.
Unknown status is not idle. Retry at most twice per logical experiment only after
a recorded mitigation and fresh identity; never after the local start cutoff.
Stop affected work for invalid numerical/evaluator/causal/resource evidence.
Stop admission at budget, deadline or permission boundaries. Keep scientific
negatives and inconclusive results; do not invent a pessimistic family stop rule.

Append to existing research MD/JSONL records. Raw artifacts are immutable;
summaries are indexes. EDA figures and source tables live in workstream `eda/`
directories. Use the live marimo cell API, not file edits to the running notebook.
Maintain coverage, claims, decisions, current frontier, costs and pending jobs.

## Submission gates

Eligible ZIPs use fixed deterministic prediction identity, root solution.py,
at most 20 MB, offline Python 3.11 compatibility, no libpython DT_NEEDED,
streaming/batch and shipped-CPU parity, finite required outputs, at most 16 GB
RAM, and one-vCPU projected/measured runtime at most 52 minutes. Forecast public
scores with selection/seed/transfer uncertainty; manual public scores are distinct
evidence. Preserve the existing user worktree and exclude wunder/challenge-3.
