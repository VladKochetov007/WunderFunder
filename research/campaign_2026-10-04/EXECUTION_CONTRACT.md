# Authorized continuation: local, Kaggle GPU, and TPU

The user accepted the continuation plan and explicitly authorized implementation,
local experiments, private Kaggle compute and dataset access, and a TPU hardware
and compatibility probe. This contract supersedes the draft contract's stale
resource limits, eight-hour budget, and claims about untested experiments. It
does not alter historical evidence or authorize competition uploads.

## Question and deliverables

Improve the complete, deployable predictor beyond the recorded 0.6655 public
incumbent (E412, user-reported October 4; original baseline 0.6641), while
testing distinct mechanisms from the reviews. Public 0.6660 is
a milestone, not an inferred result or automatic end to fundamental research.
Deliver executable experiment specifications, reproducible implementations,
immutable result bundles, research ledgers and plots, a marimo notebook, and
verified deterministic submission ZIPs for the user to upload.

## Evidence and validity

Primary metric: global_wp_clip2_v1. Legacy block WP and pub143 are diagnostics.
Tier A applies to deterministic invariants, B to controlled local comparisons,
and C to public/private transfer. Existing folds are exposed, not fresh holdout.
Use fixed checkpoints, matched teacher/propensity lineage, component exclusion,
paired component bootstrap (2000 replicates, seed 271828), and separate seed
uncertainty. Do not add correlated gains or class correlations arithmetically.
Record every negative, failure, retry, scientific acceptance, and limitation.

## Implementation before fits

First implement and check the bounded inventory: experiment specs/runner,
resource admission, artifact transport, exact epoch-boundary recovery,
optimizer/loss clock controls, priority experiment entrypoints, and notebook.
Hardware probes and small synthetic tests are admitted as implementation checks.
Do not turn incomplete or unsupported entrypoints into claimed experiments.
Resource/configuration code is frozen per run and cannot be modified by workers.

## Compute and scope

- Root Challenge 2 only; leave wunder/ and unrelated user work untouched.
- Local aggregate: 12 logical CPUs, 36 GiB RAM maximum, no swap, total observed
  GPU memory <=5,000,000,000 bytes, one GPU training process. Include marimo,
  transfer, build and monitoring children. Use lower operational thresholds.
- Local budget: automatically renewable 24 GPU-job-hour blocks, including failed
  jobs; each attempt has its own timeout. Review/reallocate after each block.
- Kaggle owner vladkochetov07; private input dataset wunder-funder contains only
  train.parquet, valid.parquet, valid_mask.parquet. It is now API-accessible.
- Kaggle GPU: two T4s reported, about 14.56 GiB each. Start at <=12 GiB observed
  per GPU; stop at 13 GiB. Profile shared RAM/CPU before simultaneous workers.
  One independent experiment per GPU; matched comparisons use the same stack.
- Kaggle quota at admission: GPU 29.99/30 hours, TPU 20/20 hours; refresh reported
  2026-10-10T00:00:00. Reserve 20% for reproduction/recovery, reconcile actual
  provider charging rather than assuming dual GPUs cost one or two quota hours.
- TPU: admit one <=15-minute inventory/compatibility kernel first. Measure
  devices, per-device HBM, host RAM/CPU, topology, framework/version and operation
  support. Subsequent experiments require a bounded matched correctness check.
  Target 80% of measured device and host capacity, with a 90% stop threshold.
- No paid provisioning before explicit spending limits. No competition uploads.
- Keep data/code/models private; never include credentials in bundles or logs.
- Preserve >=20 GiB projected free local disk. Transfer dependency closures,
  not the entire 149 GiB cache. Failed transfers cannot count as valid artifacts.

## Queue and promotion

User amendment during implementation: historical negative results are controls
and context, not automatic eligibility vetoes. Prepare bounded retests on the
current strong baseline for previously negative proposals as well as distinct
new variants. Shared identical controls need not be refitted redundantly. Only
correctness, leakage, physical/resource feasibility, or current-campaign evidence
can stop a branch. Complete the code inventory and its checks before scientific
fits; hardware inventory and synthetic compatibility checks remain admitted.

P0 complete CTRL/COMBO parent/head comparison and historic-pool final combo;
P1 actual payload runtime, width, INT8/QAT/compression; P2 cadence/EMA/loss and
optimizer controls; P3 volume/fresh views/sentinels; P4 conditional cues;
P5 structural models and teachers; P6 bounded predictive credit questions.
The accepted full plan supplies the distinct controls and old negative results.
Default block shares: 50% leads, 25% confirmation, 15% structural, 10% recovery.
Screens require correctness and an informative matched contrast. Default useful
gain is +0.0005 pooled WP, or material runtime reduction with negligible score
loss. Confirm on additional folds/seeds, report intervals and class guards.

## Incidents and stop conditions

Missing monitoring, resource overage, OOM, numerical invalidity, lineage mismatch,
or failed causal/metric invariants block affected launches. Preserve the attempt,
diagnose, and retry under a new identity after correction. Do not silently change
batch size, precision, training pool, checkpoint selection, or objective.
Resume only from full compatible training state, not inference-only weights.
Stop/reframe a mechanism after repeated informative negatives. Campaign-wide
stops are user instruction, unavailable authorized resources, evaluator failure,
or final submission freeze (72h before the verified deadline).

## Team and authorization

Use isolated implementation branches and independent review. Controller owns
budgets, IDs, launch authority and synthesis; authors do not approve their own
results. Historical pause markers remain archived evidence. The new queue uses
this authorization and a new readiness record only after preflight; do not
reactivate old queues or reuse their expired receipts.
