# Plateau review frontier

## Latest execution handoff — October21

Owner root. Live fit `wnn-h512-f0-fit-r1-20261021.service`, invocationc2b99b3c2b944edaa05156537655b04c, remains healthy; six-epochCP5recipe unchanged. [12hceiling](../submission_delivery_r1/h512_f0_followup_r1/THERMAL_BUDGET_AMENDMENT_R2.json) +25minpostfitreserve fits15hbudget; sustainedthermalclock~337MHz,1.9sequences/sbybatch560. AggregateRAM5GB,zeroTaskSwap,OOMcounters0.

The stdlib supervisor `wnn-h512-postfit-watcher-r1-20261021.service` is live and conditionally admits the reviewed [seven-step postfit automation](../submission_delivery_r1/h512_f0_followup_r1/README.md#latest-runtime-and-automated-handoff). Next after actual postfit evidence: evaluate [actualTL3replacement endpoints](../submission_delivery_r1/tl3_h512_endpoint_screen_r1/README.md). No new trained ZIP. [Kaggle terminal packet](../../campaign_2026-10-21/kaggle_h512_profile_r1/README.md) preserves two setup failures, no scientificupdates; corrected boundedwarmcohort proposal owned bykaggle_delivery_readiness, no third launch admitted. H512 prior evidence remains12matchedpairs/fourfolds, localcurrentoneexposedcell.


Updated: 2026-10-20T22:19:10.294283+00:00. Owner: root/controller.

## Resume instruction

Read the [accepted spec](../../../specs/plans/2026-10-20-plateau-review-local-kaggle.md)
and latest [project state](../../../specs/current-state.md), then inspect live
workers and Git status. Continue the first unfinished gate without reopening the plan.
Record changes and receipts here before context compaction or handoff.

## Actual stage

- [H512 TL3 packaging route](../submission_delivery_r1/tl3_member_packager_r1/README.md) now passes an engineering-only17.1MBZIP build and Python3.11native finite/warmup/reset checks. Alternating two-group/three-repeat cost is1.303×originalTL3; this is not official runtime or model quality. Do not submit the weak historical fixture.


- **Cost caveat:** the healthy H512 fit is thermally throttled at 89°C / roughly400MHz. Short-profile4.7h estimate is optimistic; use sustained epoch timings before any final-fit admission. Optional physical cooling requested, no settings changed. [Thermal record](../submission_delivery_r1/h512_f0_followup_r1/README.md#sustained-thermal-limitation).


- **H512 fit running** since23:18:35UTC under `wnn-h512-f0-fit-r1-20261021.service`, invocationc2b99b3c2b944edaa05156537655b04c. [Frozen fit/score/start](../submission_delivery_r1/h512_f0_followup_r1/README.md) admits one cold6epochF0seed1003 fit now, no teacher/final fit. PrimaryCP5 plus fixedSWA345 and75/25blends declared before training. Root owns measured timing, evaluated exports and final-training admission. Parent now5GBRAM/twoCPU/noSwap.
- TL3plain-member CPU conversion and full/chunked synthetic output/state transport pass; see [actual receipts](../submission_delivery_r1/tl3_member_adaptation/README.md#actual-cpu-conversion-and-transport-acceptance). A single-member H512replacement packager is being prepared, no candidateZIP or quality claim.


- Completed [F0 results](../submission_delivery_r1/nll_adaptation/RESULTS_F0_R1.md): best fixed oldZ4/WMSE blend0.6835218031, +.00035747 versus oldZ4; NLL−WMSE−.00004361. Broad conditional intervals, one exposed fold/seed; actual independent review pending. No new ZIP. Prepared NLLF1pair deferred for opportunity cost.
- Completed [H512 local profile](../submission_delivery_r1/h512_replication_profile/README.md):40finite updates,1.893s/batch,3.602GBpeak,zero swap. Approx4.7h fold fit and~5h production projection; exact fitting/comparison plan in preparation, no H512 fit admitted. Completed local GPU job-time~2.54h plus15.17sprofile, within15htotal.


- Both WMSE/NLL fits completed successfully in8640.57seconds total,14920updates each, zero nonfinite gradients/skips and matched sampling/normalization in both epochs. Initial export path-binding failure occurred before predictions; corrected R2 preflight passed and WMSE export is running under `wnn-overnight-nll-postfit-r2-20261021.service` / `wnn-overnight-nll-export-r2-20261021.service`. Read the [terminal/repair record](../submission_delivery_r1/nll_adaptation/README.md#both-fits-completed-corrected-export-attempt). No quality result yet. Older running-fit/R1 unit statements below are historical.

- The TL3 plain-member converter is integrated and independently statically reviewed; actual CPU conversion waits for the running fit/export/grade chain to drain. It retains unknown ancestry and is not a production training validator. A [conditional H512 cost profile](../submission_delivery_r1/h512_replication_profile/README.md) preserves the B300-supported fallback with an explicit source adaptation and no unsupported timing promise. Free Kaggle quota was refreshed:5.02unreserved GPU-hours, zero provider-reserved time; no cloud job admitted.

- [Conditional F1 comparison](../submission_delivery_r1/nll_followup_f1/README.md) now freezes seed4 WMSE/NLL commands and the exact two-endpoint scorer. It is NOT_ADMITTED until F0 is graded and resource/ancestry preflight passes. The [TL3 member adaptation](../submission_delivery_r1/tl3_member_adaptation/README.md) identifies a three-model production path by replacing only the0.34plain student; no conversion or fitting yet. Original ZIP/member hashes are verified and unknown ancestry remains explicit.

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

New-cycle local GPU time spent: **41.30seconds completed profiles +4249.17seconds completed control**, plus the running NLL fit; allocation: **15h total**. Sustained throughput revises the pair projection to roughly2–2.5hours, superseding the short-profile84minute estimate; hard reservation3hours. No Kaggle worker admitted. Inspect the actual service/receipt before relying on this snapshot. Root owns admission,
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
