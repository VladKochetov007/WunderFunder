# Five-point training repair, 8 November 2026

Status: implementation in progress; no new fits or GPU measurements.
Objective mode: engineering. Tier A tests verify software contracts; empirical
score benefit remains unmeasured (exposed official valid, no fresh holdout).

## Scope and order

1. Honest train/valid diagnostics: fitting-only representative monitoring
   cohort, explicit source/mask/comparability; unchanged full official primary
   validation. Train has no is_scored; never invent it or infer time from seq_ix.
2. Observability: distinct activation/gradient RMS and named missing gradients;
   preserve training gradients; expose objective components and native head,
   reference and mixture distributions in the local dashboard.
3. Checkpoints/plateau: retain recoverable epoch endpoints, separate fixed
   endpoint from exploratory best/SWA, safe identity/RNG-aware resume. Existing
   completed results remain historical; no automatic early stop/promotion.
4. Objectives: standalone labels are the supported new-run baseline; make legacy
   in-sample mixture/correlation assumptions explicit and prevent misleading
   OOF claims. If window correlation is implemented it must be exact, bounded
   and regression-tested, never mean-of-microchunk relabeled as global.
5. Optimizer/context: actual per-group update-to-weight telemetry and separate
   LR controls; explicit forward context, TBPTT gradient horizon and cache
   approximation; future 512/A2 and 1024/A1 require actual memory/throughput gates.

Finish and commit each point before starting code changes for the next.
Two runtime branches preserve their established module layouts. One L4/UI actor
and one B300 actor own disjoint worktrees; a bounded critic reviews independently.
ROOT integrates, runs aggregate checks and commits; agents never publish or use
SSH/GPU/auth. Native Luna/max, at most three subordinate roles, no child swarms.
CPU budget: two aggregate threads, at most one per actor, no model fits. Use tiny
synthetic correctness cases, no new packages/downloads unless actually required.

## Validation invariants and baseline

ADR0044: official train 10,325 whole sequences only -> full official valid 1,873
sequences / 3,528,273 is_scored rows. Fit all preprocessors/initializers/teachers
on fitting members only. IDs are shuffled, not timestamps. Valid is exposed.
Historical clean TL3 WP=.6923774248169172. Recent label-transformer fixed 90/10
mixture mean=.6924925011946058 across two seeds; no public improvement established.
Fixing telemetry does not by itself improve predictions. No claim of top score
or GPU speedup without new measured fits and reproducible serving evaluation.

## Evidence

Local independent audits: artifacts/training-diagnosis-20261108/recipe-audit.md,
monitor-audit.md and DIAGNOSIS_AND_NEXT_STEPS.md under Documents/Codex.
Historical bases: L4 96823b7aa; B300 14fe0bded. Never modify frozen source archives.
Tests must cover unchanged primary masks/coverage, observational parity and new
fail-closed contracts. Run Ruff and check_training_structure on changed modules.

## Progress

| Point | Status | Evidence and next-run interpretation |
|---|---|---|
| 1 | implemented; CPU checked | POINT1_L4.md; 21 tests passed, 1 dependency skip; 22 plot artifacts rendered separately, Ruff and structure 154/0; independent role regression passed. |
| 2 | implemented; CPU checked | POINT2_L4.md; 37 tests passed, 1 dependency skip; independent JS guard 41 groups/21 keys, actual full B3280 journal rendered (12 files, no inference); Ruff and structure 156/0. |
| 3 | pending | |
| 4 | pending | |
| 5 | pending | |

Each point adds a receipt with rationale, files, exact checks, limitations and
operational thresholds. GPU memory/time/quality remain unmeasured until admitted
profiles and paired fits; preserve failed checks with their correction.
