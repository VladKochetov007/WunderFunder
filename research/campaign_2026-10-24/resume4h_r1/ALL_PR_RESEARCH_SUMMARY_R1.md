---
kind: review
status: complete
created_utc: 2026-10-24T11:40:36Z
updated_utc: 2026-10-24T14:00:55Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - pr_audit_01_24/REPORT.md
  - pr_audit_25_40/REPORT.md
  - pr_audit_25_40/PR33_ARCHIVE_RECOVERY_R2.json
  - ALL_PRS_R1.json
  - feasibility_scout/RESULTS_R2.md
  - REMOTE_ARTIFACT_RECOVERY_R1.json
  - feasibility_scout/clock_eda/CLOCK_GEOMETRY_RESULT_R1.json
---

# Pull request research and artifact summary: #1–40

**Current snapshot:** a final metadata-only `gh pr list` refresh after the
13:02 UTC R2 snapshot returned the same 38 pull requests, numbered #1–15,
#17–18 and #20–40; #32 and #40 are merged, the other 36 remain open. All head
OIDs and states are unchanged. Only PR32's `updatedAt` advanced, to
13:40:06 UTC, when its existing progress comment moved from 21,745 to 23,725
optimizer updates; the comment still reports zero of four evaluated fits and
no official score. No artifact destination was added. See the [R3 inventory](ALL_PRS_R3.json),
[structured comparison](PR_REFRESH_R3.json), and
[refresh interpretation](PR_REFRESH_INTERPRETATION_R3.md). The audit made no
PR changes or automatic local merge.

## Complete inventory

The linked titles go to GitHub. Raw `gh pr view` records, full comment bodies,
changed-file metadata and detailed artifact notes are preserved in the
[PR #1–24 audit](pr_audit_01_24/REPORT.md) and
[PR #25–40 audit](pr_audit_25_40/REPORT.md).

| PR · title | State · updated UTC · exact head | Latest scientific status and artifact availability |
|---|---|---|
| [#1 · Causal Transformer with output autoregression](https://github.com/VladKochetov007/WunderFunder/pull/1) | OPEN · Oct 17 · `87b9b3ea396ff3bd4685d51420405a293a411db0` | Teacher proposal and primitives only; no profile, fit, score, or weights. |
| [#2 · Cold GRU width versus depth at matched compute](https://github.com/VladKochetov007/WunderFunder/pull/2) | OPEN · Oct 17 · `d3ed4c94c6415bcae54c1b24eb9115c4bc9a504c` | Three fixed F0 grades exist; H384×2 is +0.00191 vs H256×2, one exposed seed; body’s five-screen claim is unreconciled. |
| [#3 · GRU versus Transformer teachers for the same GRU student](https://github.com/VladKochetov007/WunderFunder/pull/3) | OPEN · Oct 17 · `689f4f19db9a5257edd595bb0f9e60e53f36e414` | Four-arm teacher/student design only; no new teacher fit, cache, or grade. |
| [#4 · Per-field ages versus seven group clocks](https://github.com/VladKochetov007/WunderFunder/pull/4) | OPEN · Oct 17 · `cc0cc8c5f25d7dd8df0de136cc01acdb5da3fe4d` | Field arm had a 50-update profile only; existing group-age vs no-age delta is near zero; no F−G grade. |
| [#5 · Function-preserving GRU widening](https://github.com/VladKochetov007/WunderFunder/pull/5) | OPEN · Oct 17 · `39e0fefbb344f7caea7463a7861f4ecbed99f00a` | Widening parity is engineering evidence; no widened fit/grade. Warm continuation is not a width effect. |
| [#6 · Frozen-parent output anchor](https://github.com/VladKochetov007/WunderFunder/pull/6) | OPEN · Oct 17 · `06fd0f1290a07ad04aeea794a645b7534dfc7b76` | Bounded 50-update profile only; no S full fit, export, or grade. |
| [#7 · Learned causal initial-state family](https://github.com/VladKochetov007/WunderFunder/pull/7) | OPEN · Oct 17 · `95272e102ef61a0852f084d0af47aa796a8817e2` | Static recipe proposal; no preflight, profile, fit, or checkpoint. |
| [#8 · Grouped feature encoding for causal Transformer](https://github.com/VladKochetov007/WunderFunder/pull/8) | OPEN · Oct 17 · `017e94bd636943afc020a8904dea907838269464` | Implementation starter sharing #1/#3 primitives; no profile or model result. |
| [#9 · Past-only smoothing while retaining raw inputs](https://github.com/VladKochetov007/WunderFunder/pull/9) | OPEN · Oct 17 · `a972168f5af24060fc053f569abcf5433b251de7` | Helper present; no fit or score. #17’s different Kalman/state screens do not test this arm. |
| [#10 · Trailing geometry and circle residual feature](https://github.com/VladKochetov007/WunderFunder/pull/10) | OPEN · Oct 17 · `d8f14f583793ceacb1f15dae32f19a072b98e941` | Idea/diagnostic only; field semantics unresolved, no profile or score. |
| [#11 · Permitted Chronos or TimesFM teacher transfer](https://github.com/VladKochetov007/WunderFunder/pull/11) | OPEN · Oct 17 · `8a0a1f100d171a3779bc4bb06d91e34bda649541` | Proposal only; checkpoint, license, input semantics, and adaptation unresolved; no reusable asset. |
| [#12 · Expand clean heterogeneous recurrent teacher cohort](https://github.com/VladKochetov007/WunderFunder/pull/12) | OPEN · Oct 17 · `6a32fc40ca962f9d45aeaf387f81ed4c546b8eaa` | One clean F3 teacher and fixed blend show exploratory +0.000875 vs older endpoint; interval crosses zero. A 299.7 MB prediction is named in private storage, not verified here. |
| [#13 · Fixed checkpoint trajectory averaging](https://github.com/VladKochetov007/WunderFunder/pull/13) | OPEN · Oct 17 · `3b6ddd1c8c53f28d96c719482c06d40eb7a56076` | Two exposed F0 seeds average +0.0000505; both share one teacher cohort. Weights remain ignored/private. |
| [#14 · Causal innovation residuals beyond moment features](https://github.com/VladKochetov007/WunderFunder/pull/14) | OPEN · Oct 18 · `3c534a41fe26bdb61eb6b66f5da94965148e12db` | Matched fit blocked by cache compatibility; proxy ridge results test neither the proposed innovation fit nor a student interaction. |
| [#15 · causal-geometry-discrete-b300: Causal geometry and discrete state features on B300](https://github.com/VladKochetov007/WunderFunder/pull/15) | OPEN · Oct 19 · `e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e` | Partial exposed matrix: 64/120 fits graded, incomplete arm means below baseline; no significance. Private evidence/weights paths are not mounted. |
| [#17 · Causal smoother GRU teachers and distillation on B300](https://github.com/VladKochetov007/WunderFunder/pull/17) | OPEN · Oct 18 · `588b93f9c2ad7e53aac7b1e2a39f30c32047ea22` | Small local screens are near zero and imprecise; no clean teacher/student cache or B300 quality endpoint. |
| [#18 · Causal predictive-state GRU with next-input self-supervision](https://github.com/VladKochetov007/WunderFunder/pull/18) | OPEN · Oct 18 · `b5b0ea0e87318ad7feba149e56f328809a7e9057` | Objective and runner code only; no profile, fit, grade, or task artifact. |
| [#20 · Meta-trained causal online adaptation for GRU teacher and student](https://github.com/VladKochetov007/WunderFunder/pull/20) | OPEN · Oct 18 · `96a98a6963af14b88187f713e593cc1421a60859` | Primitive/synthetic profiler only; no task-data fit, score, or launch-ready packet. |
| [#21 · Causal group-masked pretraining for a GRU teacher](https://github.com/VladKochetov007/WunderFunder/pull/21) | OPEN · Oct 18 · `28ff7499c7e1363a92f7599c6c68d58de966d8c2` | Code and 90-endpoint plan; profile, pretraining, supervised fits and grades absent. |
| [#22 · Adaptive TBPTT with fixed optimizer cadence](https://github.com/VladKochetov007/WunderFunder/pull/22) | OPEN · Oct 18 · `67de32b65ae3d178a0b426a10dbcbbf76717b4cf` | Controller/trainer unimplemented; no profile or fit; `launch_ready=false`. |
| [#23 · Current-superbatch WP gradient replay](https://github.com/VladKochetov007/WunderFunder/pull/23) | OPEN · Oct 18 · `53de238ea00638e591acd76b25bd59ae7fecf0cf` | Proposal only; older pooled-gradient negatives are related but not this replay recipe. |
| [#24 · Four-arm GRU, Transformer and Mamba-2 handoff](https://github.com/VladKochetov007/WunderFunder/pull/24) | OPEN · Oct 18 · `954f789b3c734d24468fb5cbba7ddfe7484cb8d0` | Historical one-epoch screen is explicitly unrun; no runnable current profile, fit, or grade. |
| [#25 · E-B300-R01: cold GRU width versus depth at fixed budget (CP30)](https://github.com/VladKochetov007/WunderFunder/pull/25) | OPEN · Oct 19 · `967f2f342e919e69f4f6cee39373b3b01b25c16a` | 12/30 paired cells (24 endpoints) receipt-verified: wide 0.684091 vs deep 0.683314, partial. Private artifacts lack public payload URLs/sizes. |
| [#26 · CP30 cold GRU: per-field vs grouped causal age (PR4 successor)](https://github.com/VladKochetov007/WunderFunder/pull/26) | OPEN · Oct 22 · `56a81f4aacaecea05df52133c9d8ccf19e055908` | New short screen reports 12 accepted/scored fits; mean deltas −0.000350 at 10% and −0.000085 at 25%. Private outputs unavailable locally. |
| [#27 · CP30 cold GRU: causal innovations vs matched moments (PR14 successor)](https://github.com/VladKochetov007/WunderFunder/pull/27) | OPEN · Oct 22 · `b8ae611da1bb80aaf6dceb73dee4b6149e8f633d` | New short screen reports 12 accepted/scored fits; mean +0.000946 at 10%, −0.000029 at 25%. No local score payload. |
| [#28 · Teacher scale × GRU capacity transfer frontier](https://github.com/VladKochetov007/WunderFunder/pull/28) | OPEN · Oct 20 · `175ccb3c8e00adc91451205c9ca9cb720f426d44` | Six-fit design only; no receiver profile, fit, grade, or teacher artifact. |
| [#29 · Causal teacher hints beyond output distillation](https://github.com/VladKochetov007/WunderFunder/pull/29) | OPEN · Oct 20 · `b32b3828254f1ab01b61446395629b8e69d09b48` | Four-fit design depends on #28 caches/controls, which are not available as validated payloads. |
| [#30 · Streaming Transformer students versus GRU with matched KD](https://github.com/VladKochetov007/WunderFunder/pull/30) | OPEN · Oct 20 · `80103720c03853dfdade18c96aeb8a08537fca29` | Four-fit student proposal; no profile, teachers, predictions, or weights. |
| [#31 · Smarter causal teachers: empirical scaling, optimization and dynamics](https://github.com/VladKochetov007/WunderFunder/pull/31) | OPEN · Oct 19 · `5d701100a28b81e7d6f4352f5f05c5fc87010d6d` | Architecture/calculator and draft recipes; no executable DDP trainer, reservation, profile, or teacher artifact. |
| [#32 · B300 BF16 teacher search and ensemble distillation](https://github.com/VladKochetov007/WunderFunder/pull/32) | MERGED · Oct 24 11:30 · `e4b3e92bd49307c1c809000f1a1474d98fb828b3` | Four fresh teachers continue under BF16 + cuBLAS BF16x9 emulation; no endpoint grades. Recovered 546-file archive SHA `743fafef6f2ee5a02c0a531c459f60d92ccb39e7c36baa323b98431df51cb36b` contains no weights/predictions. |
| [#33 · Run the local/Kaggle mechanism cycle and recover saved endpoints](https://github.com/VladKochetov007/WunderFunder/pull/33) | OPEN · Oct 23 · `4e3c4023a34595bb3dcde28f4513376581401a09` | Fixed i1 blend gains +0.000315 across two exposed cells; local i1 code, recipes and grade exist. The uploaded snapshot archive is an operational/input-bundle snapshot, not model weights. Native 367-input serving, output parity and final package remain unavailable. |
| [#34 · Integrate Harness² into the research handoff workflow](https://github.com/VladKochetov007/WunderFunder/pull/34) | OPEN · Oct 22 · `bbcb568793b408ecdf1e448cd58ca4c5eb3eef58` | Tooling CI reports 31 pass/2 skip and cross-platform checks; no ML fit/artifact. |
| [#35 · Run matched numerical-input Transformers on B300](https://github.com/VladKochetov007/WunderFunder/pull/35) | OPEN · Oct 22 · `eed60073e8b1a656fe91dd397498b2b8843163ad` | Six profiles and six fits reported in progress; no completed-epoch receipt or official score. Private outputs not mounted. |
| [#36 · Run matched mask-aligned readout fits on B300](https://github.com/VladKochetov007/WunderFunder/pull/36) | OPEN · Oct 22 · `716dbde5451726c0d01790638fded6d23a99de05` | Not admitted; exact parent checkpoints/provenance absent, zero new fits. |
| [#37 · Cold-label GRU NLL vs WMSE: four fits, three epochs](https://github.com/VladKochetov007/WunderFunder/pull/37) | OPEN · Oct 22 · `52d4e34ab618d837b8c9b6ff52057e7591f5abed` | User-paused recovery; four H256 epoch-0 states saved, no completed comparison/grade. Checkpoints remain private. |
| [#38 · Online input relationships with a frozen-parent residual head](https://github.com/VladKochetov007/WunderFunder/pull/38) | OPEN · Oct 23 · `1d389ddf5a1f195c7f0015d9a716674cbe9a0639` | Later terminal receipt supersedes old zero-fit deferral: 7 heads exported, 1 interrupted, 4 unstarted. Separate F0 grade finds all six corrections below parent; local arrays exist. |
| [#39 · TL3 student: labels vs clean-fold distillation](https://github.com/VladKochetov007/WunderFunder/pull/39) | OPEN · Oct 23 · `c1f55a167d5c0e295982dc66d3f17e95cddc617a` | Eight-fit clean-fold proposal; bundles do not exist; no profile, fit, or score. |
| [#40 · PR32: measure Blackwell precision and batch throughput](https://github.com/VladKochetov007/WunderFunder/pull/40) | MERGED · Oct 24 09:03 · `d0c6a4a1b90f1f7cebf06c24c815811ed78ae59d` | Engineering only: 10 batch candidates profiled; B96 for 630M and B64 for 1.54B selected. No quality score or scientific payload. |

No PR #16 or #19 exists in the fresh inventory; both REST and GraphQL lookups returned 404 in the first audit. PR #18 references issue #19 as a worker-claim tracker, not PR #19.

## What changed the evidence picture

**PR26/27 supersede the old zero-scored-pairs snapshot, not the old CP30
contracts.** The Oct 22 short-screen completed 24 reported new fits: 12 age
and 12 innovation attempts, each mechanism tested at nested 10% and 25%
fitting-data fidelity over F0/1201, F2/1202 and F3/1203. Both panels used the
cold H512×1 labels-only source `a266b036b02e8e012c6eb31072291dad618a169d`
with strict FP32. The PR comments report all attempts integrity-accepted,
exported and officially scored. Age was
slightly negative at both fidelities. Innovation was consistently positive
at 10% but approximately neutral at 25%. These are exposed folds with repeated
fold/seed cells across fidelities, so they do not estimate training-seed
variation or constitute independent discoveries. The previous CP30/full-data
matrices remain incomplete. The minimum four-fit cross-fidelity panel is
estimated at 6.18 hours of serial single-GPU wall time, derived from reported
B300 per-fit process timings; GPU utilization was not measured. The p10 pair
alone is estimated at 1.66 hours and p25 at 4.52 hours. Reusing exact
historical controls still leaves about 3.02 serial hours for the p10/p25
candidates, so the faithful comparison exceeds a two-hour wall allocation.
Kaggle throughput remains unknown. The source-checked feasibility note is
[here](kaggle_age_r3/FEASIBLE_NEXT_SCIENCE_R1.md).

**B300 work is active, but its current value is engineering.** #32's latest
comment reports four fresh 30-epoch teachers (630M and 1.54B on F0/F2), 47,100
planned optimizer updates, and a 34.85-hour fit estimate before setup,
checkpointing, export, and evaluation. The amendment uses BF16 model compute
with cuBLAS BF16x9 emulation and FP32 masters, gradients, loss, and optimizer
state. Width and batch change together. This is neither strict-FP32 numerical
identity nor a quality comparison. #40 measured batching and memory, not model
quality. Root recovered the separate #32 evidence archive from its fetched
branch and verified all 546 files (68,978,360 expanded bytes; archive SHA
`743fafef6f2ee5a02c0a531c459f60d92ccb39e7c36baa323b98431df51cb36b`); the
source manifest is [PR32_ENGINEERING_MANIFEST](pr_audit_25_40/remote_sources/PR32_ENGINEERING_MANIFEST.json).
The recovered tar is at `/home/vlad/development/wunder/research_tmp/resume4h_20261024/remote_artifacts/engineering-1792802811.tar.gz`;
it contains engineering profiles/logs and zero scientific fit, weight,
checkpoint, or prediction payload.

**Seven fixed online endpoints now have a root grade.** The new
[RESULTS_R2](feasibility_scout/RESULTS_R2.md) grades the frozen parent plus six
F0/seed1001 heads, with zero new fits. Parent WP is 0.6811541494; all six
head WPs are lower (best 0.6806097840). Four prespecified between-head
contrasts have conditional component intervals crossing zero. This is fixed
prediction evidence on 468 exposed sequences/141 components/808,863 scored
rows, not seed/fold uncertainty, inference-parity evidence, or a kill of the
wider direction. The grade took 32.71 seconds with zero new fits; peak memory
was 399,998,976 bytes under a 400 MB ceiling, with 6,001 `memory.events:max`
increments but no OOM, OOM kills, or swap. The execution record describes
little memory headroom and substantial reclaim pressure. It does reduce the
priority of blindly repeating the same correction recipe. Fit-residual or
objective mismatch, or a uniformly applied shrinkage diagnostic, remain
decision-relevant questions. The original F0/F1 12-fit panel is still
incomplete.

**Two earlier interpretations are corrected.** The F0/seed1003 local wide
endpoint at 0.6737706292 matches the B300 grade 0.6737706305 within
1.3×10⁻⁹. The reported 0.68409111 was the four-fold mean, not the matching
F0 cell; the earlier “failed replication” reading was wrong. This validates
score reproduction for one endpoint, not a replicated wide-versus-deep
treatment or a stronger submission. Separately, the no-target feature-clock
scan covers 1.92M rows from 96 fixed validation sequences and identifies 65
exact event classes. Seven-group ages differ from per-field ages on 14.392%
of selected scored field-age values; nine named families differ on 11.533%.
That is geometry only: it neither assigns feature semantics nor predicts a
quality gain. These details and the completed receipt are in
[CLOCK_GEOMETRY_RESULT_R1](feasibility_scout/clock_eda/CLOCK_GEOMETRY_RESULT_R1.json).

## Artifact, ancestry, and comparison limits

The broad audit found zero repository releases/assets and zero GitHub Actions
artifacts, including the eleven workflow runs associated with PR checks.
Consequently a private `/data/epanov_s/tabdata/...` path in a comment is not a
downloadable artifact. #12's 299,680,128-byte F3 prediction is a private
storage lead, not a locally verified payload. #15/#25/#26/#27 private archives
also lack a mounted path and published object size/expiry. #38 is the main
exception: seven local scratch outputs were present by stat, but no `.npy`
payload was opened in the independent review; the later root grade used its
own pinned preflight and stable target archive. #33's 2,887,016-byte uploaded
snapshot archive is a delivery/input-bundle snapshot, not trained model
weights or a final submission. Local i1 research files and the six-fit grade
remain in [the i1 packet](../../campaign_2026-10-21/next_cycle_r1/i1/README.md).
Native 367-input serving remains unavailable: the [compatibility review](../../unified/reviews/I1_DYNAMICS_NATIVE_SERVING_COMPATIBILITY_REVIEW_20261022_R1.md)
finds that the 367-slot checkpoint is rejected by the current native input
layout gate; export and callback parity/runtime checks remain unimplemented.
#32's verified engineering tarball is engineering-only. None of these
availability labels should be upgraded to “recovered model” without a
verified payload and lineage.

The comparison unit is also easy to overstate. Fixed-endpoint component
bootstraps in #12/#13/#38 quantify conditional evaluation variation, not fit,
seed, fold, or public-transfer uncertainty. The #26/#27 10% and 25% subsets
reuse the same three cells; their deltas cannot be added, pooled as independent
replications, or compared with full-data CP30 results. #4's group-age versus
no-age score is not #26's field-age versus grouped-age contrast. #15's B300
geometry mask results are not the width/depth result copied into #25. Profiles
and numeric input throughput in #32/#35/#40 do not establish scientific
quality; BF16x9 emulation must not be labeled strict FP32. The #38 chronology
is especially important: an earlier `EXECUTION_RESULT_R2` deferred the panel
at zero fits, while the later `PANEL_TERMINAL_R1` records seven saved heads;
the latter is the operative execution history.

All remote heads have read-only `origin` refs matching the fresh GitHub OIDs;
the first-range audit independently checked all 22 existing refs, and root
fetched the later PR heads. Root also fetched the separate PR32 engineering
evidence branch for hash verification. No PR branch was merged into the local
research base by this audit. GitHub reports only #32 and #40 merged; the other
36 remain open.

## Priority questions for the active window

1. **Can saved online corrections be improved without new fits?** First
   explain the common parent loss using fit-versus-evaluation residual
   mismatch, objective/regularization mismatch, and a frozen, uniformly
   applied shrinkage diagnostic. Do not reopen the old all-fold panel by
   silently changing its protocol.
2. **Does innovation value fall as fitting data grows?** The 10% positive / 25%
   neutral split is the largest promising scientific delta in new PR comments.
   A next comparison needs a complete matched cell, honest projected cost, and
   no claim of seed certainty from three reused cells.
3. **Does i1 history add beyond a strong fixed endpoint?** #33's fixed blend is
   small but positive in two cells; its history arm alone loses. Any follow-up
   should grade the whole served endpoint and preserve exposure/ancestry.
4. **Can ongoing teachers complete before the window closes?** The #32 R4
   trajectory is projected well beyond the four-hour cycle. Prioritize safe
   checkpoints, source identity, predictions, and terminal receipts; do not
   read profile activity as teacher quality.
5. **Are the clock differences useful?** The geometry result shows a concrete
   7-vs-9 grouping difference but no predictive evidence. Canonical age input
   already uses capped `log1p`; a fresh proposal must change a verified
   component and be deduplicated against existing event-clock work.

The active resume stops new admission at 14:23:12 UTC and collection at
14:38:12 UTC. All folds are exposed. A score, artifact, or successful
execution on a PR does not by itself make it a clean holdout, a winning
submission, or a reason to merge its branch.

## Source records

- [Fresh all-PR inventory](ALL_PRS_R1.json), [resume contract](RESUME_R1.json)
  and [active frontier](FRONTIER.md).
- [PR #1–24 raw records and report](pr_audit_01_24/README.md) and
  [PR #25–40 raw records and report](pr_audit_25_40/README.md).
- [PR33 archive recovery receipt](pr_audit_25_40/PR33_ARCHIVE_RECOVERY_R2.json)
  records the uploaded operational snapshot's manifest-verified recovery
  without extracting or executing archive members.
- [F0 online grade result](feasibility_scout/RESULTS_R2.md), its
  [independent review](f0_grade_review/RESULT_REVIEW_R1.md), and
  [artifact recovery receipt](REMOTE_ARTIFACT_RECOVERY_R1.json).
- [Clock EDA](feasibility_scout/clock_eda/README.md) and
  [completed geometry receipt](feasibility_scout/clock_eda/CLOCK_GEOMETRY_RESULT_R1.json).
- [Final R3 PR inventory and refresh](ALL_PRS_R3.json),
  [structured differences](PR_REFRESH_R3.json), and
  [PR32 progress/comment interpretation](PR_REFRESH_INTERPRETATION_R3.md);
  only an existing progress comment changed, without a new quality endpoint
  or artifact link.
- [PR26/27 next-science feasibility](kaggle_age_r3/FEASIBLE_NEXT_SCIENCE_R1.md)
  and [event-clock literature note](pr_audit_25_40/clock_literature/README.md).
