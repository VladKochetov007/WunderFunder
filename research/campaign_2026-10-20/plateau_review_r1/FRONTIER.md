# Plateau review frontier

Updated: 2026-10-20T22:19:10.294283+00:00. Owner: root/controller.

## Resume instruction

Read the [accepted spec](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and latest [project state](../../../specs/current-state.md), then inspect live
workers and Git status. Continue the first unfinished gate without reopening the plan.
Record changes and receipts here before context compaction or handoff.

## Actual stage

- Latest October20 steering after failed SSH discovery: train the small GRU
  locally. The [delivery packet](../submission_delivery_r1/README.md) now prioritizes
  a matched H256×2 WMSE/NLL continuation pair using local weights and cached
  teacher targets. Beleriand access is no longer a prerequisite for progress.
  Both profiles passed; the two-fit sequence is running as `wnn-overnight-nll-fit-20261020.service`.
  Control completed both epochs in4249.17seconds with14920successful updates,
  zero nonfinite gradients and zero skipped updates. NLL is running; its
  progress and the control receipt are indexed in the delivery packet. The durable
  `wnn-overnight-nll-postfit-20261020.service` waits for exact pair success, then
  executes the frozen three exports, artifact audits and seven-endpoint score
  with a45-minute cap. No fitting quality result is claimed. See the packet for
  exact admission, actual progress and serving preparation.

- October20 latest execution priority: [overnight submission delivery](../submission_delivery_r1/README.md), evaluating saved FG19 control/EMA first and admitting useful bounded training if needed. The user allows all-night compute; root preserves the local 15h cap and resource limits. Original i1 comparison is deferred, not cancelled or running.

- Latest teacher-first steering is preserved in the [scaling programme](../../unified/ideas/TEACHER_QUALITY_SCALING_20261019_R1.md): optional distillation, six-fit ladder proposal and ranked tuning. No new fit/profile or budget admission; local plan remains preserved.

- Later October19 steering makes4–6 TOTAL new fits the default initial screen
  for future drafts and prioritizes large B300 Transformer teachers→GRU transfer.
  [ADR0043](../../../specs/adr/0043-fast-screens-and-transformer-teacher-scale.md)
  and the [teacher design](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
  preserve this direction, its independent review and unimplemented distributed
  training path. Existing B300 results were collected in the linked PR review;
  no B300/local/Kaggle job was admitted. The accepted local plan and unused15h
  allocation remain preserved; reconcile the latest priority before spending.
  Shared policy is pushed on main/research-team; [draft PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31)
  adds separately counted teacher-scale and TradeFM-inspired event-objective
  panels. Preferred first dynamics comparison uses likelihood off/on at0.68B.
  No B300 fit, local job or teacher/student result was added by this preparation.
- Accepted plan saved and linked; no new fits or scientific scores.
- Baseline source: main `9defbcbc0f2a6f1b35941dc657496993b48889c6`. Plan/source audit
  pushed in `372af11fe`; first-comparison code integrated in `0cc795070`.
- Four user files, including `new-meta-review.txt`, are archived verbatim;
  [SOURCES.json](SOURCES.json) binds original bytes and archived destinations.
- First A/B/C comparison: cold H512×1, seed 1001, 10% complete fitting
  components, full exposed F0, 367 equal-width inputs. History/mask implementation
  is integrated with prediction/checkpoint/provenance paths. Ruff and structure
  checks pass; independent static review found no concrete first-route blocker.
  See [implementation record](IMPLEMENTATION.json) and linked review.
- [Prepared recipes](recipes/README.md) contain exact fit/predict/score and profile
  argv. Independent command-source review found no flag/path mismatch; numerical checks, complete source seal,
  actual-shape profiles and admission remain unfinished. No test or GPU job ran.
- Kaggle read-only audit found about 5.02h gross GPU quota remaining, zero provider
  reservations and no active kernels in the recent set. Refresh on admission;
  device availability and other-agent planned reservations remain distinct.
- First F0 subset is frozen: 315 fitting components, 1,118 train + 118 valid
  sequences, full 468-sequence evaluation. See `fit_f0_10pct_s1001.json`.
- The meta-review is reconciled in the spec's dated Amendment 1. Frozen-parent
  anchoring needs a fresh FP32 pair; old FP16 C is not reusable. A newer F3 wide/deep
  study belongs to another stream: collect its evidence before duplicating it.
- Model-fitting start: October 20 Kyiv. No automatic scheduled job is claimed.

## Budget and ownership

New-cycle local GPU time spent: **41.30seconds completed profiles**, plus the currently running local two-fit sequence; allocation: **15h total**. Sustained throughput revises the pair projection to roughly2–2.5hours, superseding the short-profile84minute estimate; hard reservation3hours. No Kaggle worker admitted. Inspect the actual service/receipt before relying on this snapshot. Root owns admission,
source freeze, integration and evidence handoff.

| Role | Owner and scope | Current state |
|---|---|---|
| Implementation | `plateau_i1_inputs_actor`, Luna/max; code only, no fits | Complete, actor commit `3c1936161`, integrated unchanged on main as `0cc795070`; isolated source worktree retained |
| Provider readiness | `plateau_kaggle_readiness`, Luna/max; read-only metadata, no uploads | Complete; indexed snapshot records ~5.02h gross remaining quota |
| Source/code critic | `review_direction_audit`, Luna/max; read-only source/ancestry review | Meta-review, implementation and command-source audits complete; no numerical execution |

The implementation actor may not approve its own scientific result or change
the recipe/budget. Root owns integration and later resource admission.

## Current next actions

1. Monitor the local matched WMSE/NLL pair and record actual elapsed/resident/device memory.
2. Execute the pre-outcome seven-endpoint plan: reused FP32 parent, two children,
   old-Z4 FP32 re-export, fixed50/50parent-NLL and fixed75/25old-Z4/child blends.
   Score only the468source-qualified F0 groups/141components and independently
   grade the official metric and conditional component uncertainty.
3. Form a candidate only from actual endpoint evidence, measure the native ZIP precision/runtime/size, and preserve negative outcomes. Root owns delivery.

## Deferred original i1 next actions

1. Read the saved implementation/command reviews and confirm the same source.
2. Complete bounded numerical, masking, causal-prefix and checkpoint restoration
   checks before fitting; source review alone cannot establish those properties.
3. Freeze the accepted source/environment, refresh resource controls and Kaggle
   quota/ownership, profile actual shapes and reserve complete paired execution.
4. On October 20, admit bounded local A/B/C and a paired Kaggle F2 cohort only
   after their source/data/resource dependencies are satisfied.

Do not launch a broad loss/history/sampling sweep, restart the old R4 matrix,
count draft PRs as results, or delete historical stop receipts.
