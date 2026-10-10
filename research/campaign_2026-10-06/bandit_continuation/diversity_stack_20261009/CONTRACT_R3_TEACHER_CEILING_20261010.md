# Research contract R3 (2026-10-10): raise the teacher ceiling, convert to servable students, ship valid archives

Skill: conduct-creative-research. Objective mode: **competition**. Verification tier: **B** (repeated empirical measurement on four component-disjoint folds, all historically exposed; public board is a 5/day instrument with ~0.0005 noise; reserved-fold protocol in hpo/VALIDATION_PROTOCOL.md). Model-to-reality gap: test is harder than validation, loss concentrated in class 3; local-to-public order is unreliable below ~0.003 (P2 case: local +0.0015, public -0.0025).

## Question and utility
Which teacher-side change raises the achievable score and survives distillation into a student that runs on one CPU core in under ~25 minutes locally (portal showed up to 2x slowdown)? Utility = expected public gain per GPU-hour, weighted toward class 3.

## Baseline and current evidence
Public best 0.6664 (Z5, all-data H230 student + head finishing). Scratch teachers (fold 1): h256 0.6657-0.6678, h384 0.6685 (seed mean), seed-average of 2 h384: 0.6702; students 0.6683-0.6694; HPO plateau 0.6676-0.6678 (adjusted). Width 384 +0.0012 over four folds; chunk 500 +0.0020 (one run); loss lives in the sign (oracle sign 0.843, oracle magnitude 0.685).

## Falsifiers that stop a branch (per branch, not global)
- Lookahead ladder (information ceiling): if WP with 8 rows of lookahead is below ~0.75 the labels are not near-future functions of the features and a lookahead-based privileged teacher cannot add much.
- Chunk effect: not replicated by seeds 4/5 (mean gain < 0.001) -> closed.
- Teacher gain does not transfer: student distilled from the improved teacher ensemble does not exceed the matched 256-wide-teacher student on two folds -> keep teacher work as measurement only.

## Budgets and permissions
Vast lane B (account 2, cap 36 USD, ~0.85 USD/h), local GPU (one fit, loader worker 1, 5 CPU / 18 GB), Kaggle (105k GPU-s, T4s, ADR constraints), external ideators Grok 4.7 / Freebuff (GLM) / OpenCode (free models) in read-only mode. Account 1 is exhausted (5 USD, no top-up). Human gate: uploads (5/day, 3 used), any cap raise.

## Plan
1. Information-ceiling ladder (INPUT_LEAD 2/8/32/128/512 on fold 1): queued on lane B (trainer v6_lead, offline-only, packaging guard added).
2. Chunk-500 replicates and interaction (lane B), width x chunk.
3. Many-seed teacher ensemble of the best recipe + one-cache student (scale-then-distill with ensemble target noise removed) after items 1-2.
4. External ideation round 2: reframings of the task, elephants, teacher-ceiling mechanisms (Grok, Freebuff, OpenCode), then specification critics and cheap CPU tests before any training.
5. Valid archives: only candidates that pass gates (Python 3.11 at three CPU levels, ABI, package-vs-GPU agreement, local timed replay < 25 min) and have either a new mechanism with replicated gain or a documented calibration purpose; delivered under submissions/upload_<date>/.
