# ADR-0010: Keep authorized experiment delivery active

Status: Accepted, 2026-10-07. Extends ADR-0005 and ADR-0006.

## Basis

The user explicitly orders consistent experimentation and autonomous repair.
E520's two failed cloud attempts were collected, and a repair was prepared,
but every execution owner finished before anyone reviewed or launched that repair.
Available ideas and quota were left idle. This was an ownership gap.

## Decision

An admitted research owner carries a runnable arm through execution, diagnosed
repairs, checkpoints/predictions and independent evaluation. A terminal provider
failure triggers diagnosis and a concrete revised attempt within its allocation;
it does not finish the assignment while authorized progress remains available.
The controller keeps an active owner for each executable next step and prepares
the next recipe while the current job runs. Collecting logs and writing code are
intermediate milestones. A monitor alone is not a result owner.

Use existing role/tooling boundaries. Independent review stays proportional to
the changed code and current stage. Profile-only execution does not wait for a
future fit grader. Fit preparation and review overlap the current cloud profile.
The controller may explicitly delegate conditional packet admission to a cloud
operator who did not implement or approve the candidate under review. Conditions,
exact hashes, runtime and cumulative quota ceilings must be recorded.

Keep useful compute active when an authorized, safe, runnable arm is available.
Avoid filling hardware with duplicate or irrelevant jobs just for utilization.
If one lineage is blocked, progress an independent tree arm. A real safety limit,
exhausted budget or unavailable input can block admission; a historical failure,
uncertain scientific gain or routine permission request cannot substitute for
execution. No blind OOM retry or unreviewed identity/metric change is allowed.

## Current scope

New fits use Kaggle after the existing 10:00 Kyiv local cutoff. The latest local
aggregate CPU ceiling is six cores; RAM remains 20 decimal GB, VRAM five decimal
GB, and swap zero. Kaggle retains separate resource headroom and live quota checks.
E520 recovery receives a cumulative 7,200 accelerator-quota-second ceiling,
including prior 848.447 seconds, with at most 1,800 wall seconds per profile
attempt. Its scratch pair fit has separate conditional admission up to 10,200
wall seconds, accounting for up to 20,400 quota seconds on two attached GPUs.
TPU recovery receives 7,200 quota seconds with representative real-device checks
before fitting. All allocations preserve the current account reserve and campaign
availability; no paid compute or automatic public submission is authorized.

## Evidence and revisit

[E520 delivery handoff](../../research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_CLOUD_DELIVERY_HANDOFF_20261007.json)
preserves the failed attempts and their null scientific outcome.
[CPU override](../../research/campaign_2026-10-06/bandit_continuation/runtime/LOCAL_CPU_OVERRIDE_20261007_6CORES.json)
records the latest enforced local cap. Revisit allocations on measured runtime,
terminal quota reconciliation, a scientific result or new user instruction.
