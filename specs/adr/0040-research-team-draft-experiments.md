# ADR0040: research-team experiment draft PRs

Status: accepted user protocol, 2026-10-17; local preparation before publication.

## Context

The user requests a shared integration branch and one English draft PR per research idea, with code, reproducible runs and results recoverable through unified notes. They explicitly skip a queue manager and request an explanation before this delivery is pushed.

## Decision

Use main → research-team → experiment/<id>. Canonical instructions/skills remain on main; experiment code/receipts merge through research-team and a later integration PR. Require30 paired seed/fold cells per scientific arm, strictFP32 (autocast/TF32 off), explicit teacher/student/connection lineage, mechanism/falsifier, useful-effect decisions, leakage/representativeness and separated uncertainty. Profiles/parity checks are engineering diagnostics. Quantization belongs to later submission work.

The [workflow guide](../../research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md) defines branch pinning, exact primary statistic, logging, separate evidence/weights archives and unified trace. [Inventory](../../research/unified/ideas/EXPERIMENT_DRAFT_INVENTORY_20261017_R1.md) lists prepared ideas and readiness. Canonical skills live in .agents/skills and Claude accesses symlinks; CLAUDE.md links AGENTS.md.

## Consequences

Two-arm quality comparison needs60 fits. Thirty cells do not guarantee significance, fresh holdouts or new-regime robustness. Old FP16 one-fold runs remain historical evidence and do not count silently. Draft readiness is distinct from a runnable source-clean full-data B300 recipe. No model fits, rental, upload or remote publication is authorized by this preparation. The October15 pause remains for actual fits until resumed.

## Subsequent publication authority

The user subsequently explicitly authorizes gh, pushing these branches and creating draft PRs for teammates. This supersedes the initial pre-push hold, while leaving fits, rentals and billing outside the preparation scope.
