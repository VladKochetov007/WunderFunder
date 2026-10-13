# ADR0035: Official team code and weight sharing

Status: Accepted, 2026-10-13. Basis: the user confirms the official team merger, permits mutual code and weight reuse, requests the AGENTS.md update, and approves the joint-model implementation plan.

## Decision

Within team kotiki, directly reuse, modify and combine team training/preprocessing/serving/build code, native libraries, model weights, checkpoints, teachers and initializers for research and submissions. Independent reimplementation or retraining is no longer required solely because the asset originated with a teammate.

Record author, source revision or archive hash, imported and modified source, fitting membership, teacher/initializer ancestry, precision and evaluation exposure. Preserve original assets. A shared checkpoint or rerun is not another independently trained seed. Scientific controls, independent criticism, permitted data, deterministic causal serving and the official metric remain binding.

This supersedes only the team-sharing restrictions in ADR0004, ADR0012 and ADR0016 and conflicting historical prompts/skills. Keep historical experiments, claims and target-treatment decisions intact. No additional compute, paid capacity or external upload authority is created.

## Current application

Audit the supplied xteam_h4, pair_ftt2, e6a and class-masking packages against the existing C/E endpoints. Search remote branches autonomously and bind any later teammate source to exact commits. The user selected a complete research comparison before new ZIP preparation. Missing source-qualified CV artifacts do not prevent diagnostic scoring, but diagnostic fitted-data scores cannot be relabeled as held-out blend-selection evidence.

Reported team public best is pair_ftt2 at 0.6682; xteam_h4 is 0.6681 and E is 0.6675. These are user-transcribed portal results, not independently queried results. A second E upload is pending and remains a distinct event.
