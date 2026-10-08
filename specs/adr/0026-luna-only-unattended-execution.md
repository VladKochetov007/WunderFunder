# ADR0026: Luna-only delegation and unattended finite experiment queues

Status: accepted, 2026-10-08. Basis: latest explicit user instructions to move all existing subagent context to Luna6/max because premium models are too expensive, and to run experiments in the background after Codex usage ends.

All native subordinate roles use `gpt-6-luna` with maximum reasoning. The one Sol continuation exception in ADR0025 is withdrawn. Sol was interrupted; a fresh Luna continuation role received its artifact paths, pending work and binding limits. The independent grader and scoped runner builder already use Luna/max. No Sol or Astra role may be spawned or restarted under the old exception. Root retains admission, spending and synthesis.

The requested background manager runs ordinary deterministic software independently of an LLM session. It may execute only a finite, hash-bound queue of root-reviewed jobs under a bounded policy. A long-lived root receipt may authorize fresh short-lived dispatch receipts for those exact jobs after live resource, quota, dependency and source checks. It cannot invent recipes, increase budgets or blindly retry an uncertain provider launch. Repairs use separately identified, reviewed successors; valid checkpoints and healthy workers are preserved.

The manager must preserve heartbeats, phase timings, terminal logs, checkpoints, prediction identities and fold membership on local storage. It stops new admission on a stop marker, budget exhaustion or failed checks, and drains healthy admitted work. Paid instances require receipt-proven ownership, collection and teardown when useful admitted work ends. Existing five-CPU/20GB-RAM/5GB-VRAM local caps, one bounded support task, existing credit and $20 cumulative Vast ceiling remain binding; r5 stays within its $5/ten-hour reservation.

Warm cohorts reuse verified parsed inputs where measurements justify it. OS file-cache reuse is recorded separately. Full evaluation membership and causal histories remain fixed; subset screens and previously exposed folds cannot be presented as fresh confirmation. TPU efficiency is an empirical device comparison, not an assumption.

Implementation completion requires a tested manager, concrete admitted queue, installed background service and observed heartbeat/lifecycle evidence. Prepared code or a monitor alone does not satisfy the user's request.
