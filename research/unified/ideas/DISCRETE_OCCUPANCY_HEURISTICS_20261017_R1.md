---
kind: idea
status: proposed
created_utc: 2026-10-17
updated_utc: 2026-10-17
hypothesis_ids: [H-DISCRETE-OCCUPANCY-20261017-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-17/moment_geometry_r1/README.md
---

# Discrete occupancy and regime heuristics for recurrent teachers and students

## Question or claim

The user asks whether handcrafted/discrete features, including heuristics, could help both teachers and students and explicitly says an idea is not expected to beat the strongest system immediately. This direction remains eligible after a weak tiny-model screen. Utility includes standalone quality, convergence, teacher target complementarity, student transfer and interaction with existing features. No numerical gain prior is assigned.

## Mechanism and method

The [visually inspected mean/std EDA](../validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) shows recurrent narrow price branches, multiple volume clusters, dv mean/std coupling and two a7 branches. Selected near-arc raw traces spend long stretches at one frequent value and alternate with bursts of other values. This motivates distinguishing **what level is present**, **how often it appears**, **how long it persists**, and **what happens at changes**. These are hypotheses about representation; the geometry does not identify physical market semantics.

Proposed reusable feature families, each with its own ablation:

| Family | Causal representation | Hypothesized deficiency addressed | Cheap falsifier / distinguishing observation |
|---|---|---|---|
| Frequent-level identity | Fitting-only vocabulary of common raw values, current membership indicators and an unknown-level channel | A continuous projection can blur nearby numbers that have different categorical meaning | Distinct common levels have no stable frequency/regime distinction, or matched capacity numeric encodings do as well |
| Row occupancy | Expanding and fast/slow EMA occupancy of each common level | A narrow recurrent state must estimate long-history mixture proportions implicitly | Removing occupancy leaves matched recipe and learning dynamics unchanged at useful scale |
| Event occupancy | Update level counts only when the relevant field changes; keep row occupancy as a separate channel | Long held values dominate row statistics and may hide activity-clock structure | Event-clock values add nothing over existing event flags/dwell features |
| Dwell and transition state | Log capped time since a level/change, transition rate and event direction; reset at sequence boundary | Time on a level and switching frequency are different state variables | Same occupancy but different dwell/transition patterns do not change predictive residuals |
| Soft geometry regime | Signed residuals to multiple train-fitted moment branches, confidence/sample count and unknown regime channel | One scaler averages over multiple anonymous distributions | A single branch adequately describes all sampled supports; or branch information is redundant after raw/moment features |
| Level plus scale | Raw level, causal centered value, log scale and explicit scale confidence | Scaling alone may erase signal; raw-only learning may struggle with nuisance variation | Matched raw+stats+scale has no useful contribution at the tested endpoint; this does not reject other conditioning methods |

Fit vocabulary/branch/scaler constants on fitting components only, without targets. Never use final sequence means/extrema, future events or held-out group labels as inputs. Exact floating equality is permissible only for observed stable encoded atoms; test tolerance, rare levels, drift and unseen values explicitly. Count observed transitions and rows separately. Do not call an inferred category a market class without evidence.

## Teacher, student and distillation paths

1. Give a teacher a richer multi-timescale dictionary and branch description; compare heterogeneous teacher predictions and fixed target mixtures, not just teacher standalone score.
2. Give a student a compact streaming state: a few counters/EMAs/one-hot indicators plus existing event features. Persistent state and update cost should be measured before serving promotion.
3. Factor teacher-only features versus student-only features versus both. A stronger teacher need not transfer to the same narrow student; explicit student context may remove a compression bottleneck.
4. Distill predictions from causal free-running teacher endpoints with clean component ancestry. Future-derived labels may be auxiliary fitting targets, never student inputs.
5. Test interactions with existing projected causal moments, field/group age and input encodings. An individually weaker arm can still improve teacher targets, residuals or a fixed served blend.

## Evidence and interpretation

Current evidence is input-only descriptive geometry and raw traces. The first registered moment pilot combines ordinary expanding statistics, one anchor occupancy and one arc residual before separately adding normalized current values. It cannot attribute any result to a single heuristic. A later dictionary/event-clock experiment should preregister separate controls and a combined endpoint, preserving all earlier evidence.

## Limits and validity

No predictive benefit, market-class interpretation, scale law or leaderboard impact is established here. The screenshot feature remains visually matched rather than proven. Exposed folds stay exposed. A full recipe claim requires paired seed/fold repetitions and a stronger endpoint than a tiny pilot; the team's30-cell target remains a confirmation plan rather than a fictitious completed result.

Do not kill this mechanism because it does not beat the strongest model at first. Preserve negative recipe evidence and specify what change reopens allocation. A confidence interval crossing zero does not itself settle the scientific mechanism or conditional contribution.

## Decision and next step

Keep the discrete/heuristic family active. Complete the frozen first pilot, then prioritize current-level dictionary plus row/event occupancy, with matched input width and baseline shared initialization. Evaluate both a student endpoint and contribution to a teacher-derived target; record adaptation cost, convergence and serving-state requirements alongside official score.

## Source records

- [Geometry contract, figures and frozen pilot](../../campaign_2026-10-17/moment_geometry_r1/README.md).
- Existing reusable [event features](../../../src/wnn/features.py), [projected moment/age features](../../../src/wnn/research_features.py), and [modular trainer](../../../src/wnn/training/README.md). These already cover related mechanisms; comparisons should reuse and ablate them rather than claim the broad idea is entirely new.
