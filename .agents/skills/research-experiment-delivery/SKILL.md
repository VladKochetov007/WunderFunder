---
name: research-experiment-delivery
description: Own an already-authorized research experiment from a frozen runnable recipe through bounded execution, failure diagnosis, checkpoints/predictions and an evaluated artifact handoff. Use when experiments stall between code builders and compute operators; do not launch experiments for a read-only review request.
---

# Deliver a computation result

Establish the exact result owed and who owns each dependency. Distinguish code-only, resource-profile and scientific-fit scope. For an end-to-end task, implementation is an intermediate milestone; the final deliverable includes actual execution and measured artifacts.

Make a small dependency DAG for this arm. Reuse valid existing data/checkpoints/tools when the recipe permits. Unrelated missing families, exploration jobs or future ablations must not gate a runnable control. Fix the comparison/endpoint before results; preserve changed recipes as new attempts.

Verify source/environment/input membership and profile representative actual shapes within the current local/cloud allocation. CPU/RAM/VRAM caps and provider quotas come from the live user contract, not universal skill defaults. Account for simultaneous support work and pagecache. Use durable workers, bounded resource monitoring and a single shared local GPU admission authority where required.

Record phases independently: input validation, profile, fit, checkpoint, export, evaluation and package. Provider RUNNING does not establish fit progress. Capture expected/observed data identity and phase start/end before raising errors. Observe exits, OOM counters and complete checkpoint/output status. Download terminal cloud artifacts and reconcile provisional quota.

On failure, preserve the first attempt, identify the concrete cause and execute a corrected identified version within authorization. Never blindly retry an OOM, silently relax identity/evaluation checks or rerun completed science for cosmetic status. A physical encoding mismatch requires identity reasoning appropriate to the data; do not build hypothetical fallback machinery when exact permitted inputs already verify.

Coordinate independent grading/review; the actor cannot approve its own candidate or change its grader/budget. Update lineage records with executed commands, hashes, resources, outcomes and uncertainty. Save diagnostics/plots where future agents can find them.

Return the last completed stage and the actual checkpoints/predictions/report, measured costs, failures/revisions and next owner/action. If blocked, keep working on independent dependencies and name the concrete missing evidence. Honor explicit user stop/resume instructions and retain the historical record.
