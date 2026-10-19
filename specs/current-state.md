# Current project state

## Latest steering — October 19: fast screens and large B300 teachers

The user replaces mandatory30-per-arm experiment matrices for FUTURE drafts
with **4–6 total new fits per initial idea comparison**, normally2–3 matched
two-arm cells. Count cached controls and teacher/student stages explicitly;
retain historical frozen panels and expose small-screen uncertainty. The new
strategic priority is a large causal Transformer teacher on8 B300s, followed by
GRU distillation. See [ADR0043](adr/0043-fast-screens-and-transformer-teacher-scale.md)
and the [detailed design](../research/unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md).
The design proposes0.68B/1.65B capacity recipes, optional3.17B extension, strict
FP32 Muon, WSD, profiled DDP and causal dynamics objectives. It supplies no
new runnable DDP trainer, B300 reservation, fit or quality claim. Preserve the
local/Kaggle plan below and its start/allocation; do not restart stopped jobs.
The later user-supplied TradeFM paper motivates a fixed-scale event-likelihood
screen before the full ladder; its four fits and the capacity panel are separate.
[Draft PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31) contains the
source-bound design handoff and analytical tools; no runnable teacher fit exists.

## Active instruction — October 19, 2026

The user accepted the [plateau-review local/Kaggle plan](plans/2026-10-20-plateau-review-local-kaggle.md),
requested implementation, and asked that it survive context compaction. This is
the canonical plan; the [campaign frontier](../research/campaign_2026-10-20/plateau_review_r1/FRONTIER.md)
records actual progress and the next action. Preparation is active. Model fits
start October 20 Kyiv; no scheduled or running job is claimed at this stage.

Allocation: **15 local GPU-hours**, **all verified available Kaggle quota in
parallel**, no paid compute, one local GPU worker, two aggregate CPU cores,
5,000,000,000 bytes task RAM, 5 GB VRAM and no swap. The local/Kaggle cycle has
**no mandatory 30-cell matrix**; full fitting remains eligible within budget.
The first comparison is cold, equal-width i0-only / standard / causal i1-history
GRUs on matched components. The imported review's larger grids are deferred.
All existing folds are exposed. Code/recipe preparation is not a scientific result.

The first-comparison implementation is integrated on main as `0cc795070`: causal
i1 histories, post-normalization semantic masks and checkpoint/prediction support.
Ruff and training-structure checks pass, with an independent static review. Exact
commands and fixed fitting membership are prepared; numerical checks, source seal,
actual-shape profiles and execution remain pending. See the campaign frontier.

The October 19 B300 PR status review found 12 verified but partial width/depth
pairs and 16/30 geometry pairs per arm; field-age and innovation have no scored
paired cells. The tabdata jobs were explicitly stopped. See the
[PR evidence review](../research/unified/reviews/B300_PR_RUN_UPDATE_REVIEW_20261019_R1.md)
before repeating these comparisons.

This scoped authority succeeds the October 18 stop for this cycle. Keep its
receipts and unrelated campaign restrictions intact. Read the new spec before
interpreting the dated snapshots below as active instructions.

## Historical October 18 state and results

**Latest instruction, 2026-10-18 08:27 UTC:** the user stopped research admission and prohibited new jobs. The local durable stop barrier `15e691414aa14bb4a3d888b841d85ae3` drained without terminating healthy work. No new fits, scores, packages, uploads or PRs are admitted. One Beleriand F1/seed-1001 run was last observed active at 08:20:52 UTC; let it finish and collect terminal artifacts only, with no successor. The user reports both geometry ZIPs already submitted, so they are not new candidates. See the [stop receipt](../research/campaign_2026-10-18/online_features_r1/runtime/STOP_AUTHORITY_20261018_R1.json), [operations note](../research/unified/operations/RESEARCH_STOP_20261018_R1.md), and [nightly research closeout](../research/campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md).

**Snapshot date:** 18 October 2026. **Latest user scope:** deliver one newly trained exploratory ZIP by **02:50 Kyiv (23:50 UTC, 17 October)** and publish the larger study as an 8×B300 draft PR to `research-team`. The multi-day local R4 matrix is stopped and drained. The [completed quick pair](../research/campaign_2026-10-18/quick_geometry_probe_r1/README.md) scored feature 0.6804165 / zero control 0.6804784 / original parent 0.6812265 on one exposed F0; this short continuation did not improve. The [adapter-only followup](../research/campaign_2026-10-18/geometry_adapter_probe_r2/README.md) also completed negatively: 0.6804499040 versus parent 0.6812265132, conditional component interval for the difference [−0.00135282, −0.00027913]. Bitwise checkpoint checks confirm all old weights stayed fixed; 60 new columns learned nonzero weights. Both results are one exposed F0/seed, not robust/public evidence. Local fit admission is drained; no further local fits are planned. The [ready directional q16 quota ZIP](../research/campaign_2026-10-18/geometry_adapter_probe_r2/RELEASE.json) is 5,323,200 bytes (SHA955bff715…): native served F0 WP0.68045197, exact-ZIP Python3.11 and full-F0 replay pass, local row-cost projection40–45minutes with target runtime unmeasured. The predeclared Torch/native FP32 tolerance failed and is explicitly recorded; the actual native endpoint is evaluated directly. It is released under the user's deadline exception, not promoted as a demonstrated gain. The 8×B300 study is published as [draft PR #15](https://github.com/VladKochetov007/WunderFunder/pull/15), base `research-team`.

## Latest remaining-hour continuation

The user's subsequent instruction explicitly authorizes a few more variations before the unchanged02:50Kyiv deadline. The [R3 packet](../research/campaign_2026-10-18/geometry_variations_r3/README.md) admits two matched moments-only/dictionary-only frozen-parent fits after a successful actual-shape mask profile, plus a separately predeclared inference-only feature-correction scale grid. The existing ready q16 ZIP remains frozen. This supersedes the earlier no-further-fit statement only for these bounded experiments; no broader local matrix or new cloud spending is admitted. One selected endpoint may get a complete native F0 replay, based on actual633-second prior measurement. Exposed-F0/adaptive selection limitations remain explicit.

## Latest completed variation screen

All two R3 fits and four inference-only alpha exports completed and drained by23:25:55UTC. Independent grading passed all source/hash/alignment and matched-fit gates. The top point endpoint is R2 geometry columns scaled by−0.5: WP **0.6814221100**, versus original parent0.6812265132 and full R2 0.6804499040. Conditional component difference versus parent is+0.00019560,95%CI[−0.00002480,+0.00042143]; versus R2+0.00097221,CI[+0.00027186,+0.00176521]. These are adaptive exposed-F0/one-fit comparisons, not independent-seed/public evidence. Moments-only0.68086802 and dictionary-only0.68061687 both remain below parent. The selected q16 native replay passed: served WP0.6814234038,468groups/9.36Mrows,595.41s wall,820168KiBpeakRSS; exactZIP offlinePython3.11/reset/ABIchecks passed. [Release receipt](../research/campaign_2026-10-18/geometry_variations_r3/RELEASE.json) binds the5,323,608byte SHA27f01f53…ZIP. Callback-only38.09minute projection is not measuredcompetitionruntime. No further fits belong to the closed deadline packet; new continuousresearch authority below is separate.

The user-reported [R2 result](../research/campaign_2026-10-18/geometry_adapter_probe_r2/PUBLIC_RESULT.json) is **0.66428** (30m47s/626MB, ID `14s6l2z7`); [R3](../research/campaign_2026-10-18/geometry_variations_r3/PUBLIC_RESULT.json) is **0.66456** (30m32s/627MB, ID `1wmshmaz`). Both are below the reported TL3 best of **0.66832**; R3 is higher than R2 by 0.00028 but still trails TL3 by 0.00376. Portal outcomes are transfer evidence for these exact packages and do not calibrate new local variants into public forecasts.

## Historical continuous research authorization — closed 18 October

The earlier user instruction authorized autonomous overnight and longer research **until explicit pause**, including causal online learning, custom/learned features, previous-competition methods, TL3 complementarity and larger promising runnable B300 draft PRs. The [continuous campaign](../research/campaign_2026-10-18/online_features_r1/README.md) preserves that authority and its 2-CPU/5-GB task-RAM/one-GPU limits. The explicit 18 October stop supersedes its admission scope; the completed data audit, experiments, frozen sources, ZIPs and results remain evidence.

### Latest online-learning evidence

The [causal proxy gate screen](../research/campaign_2026-10-18/online_features_r1/causal_proxy_gate_r1/RESULTS_R1.md) found no useful gain from this one pseudo-label gate: best of three timescales improved fixed 75/25 by +0.00004795 WP (32-component conditional interval [−0.00003098,+0.00012853]) and remained 0.0023302 below TL3 on the same repeatedly exposed F0 components. Target maturity is undocumented and TL3 ancestry remains unknown. Prediction reconstruction passed before labels/masks were read; three CPU-only attempts had zero OOM deltas. This narrows the exact proxy-gate recipe, not online learning broadly. No separate B300 draft was warranted for this gate. The independent evidence review and execution receipt are linked from the [unified validation index](../research/unified/validation/README.md).

The [causal Kalman input screen](../research/campaign_2026-10-18/online_features_r1/kalman_filter_probe/RESULTS_R2.md) remains an inconclusive positive fixed-GRU diagnostic (+0.000338 WP for local trend versus raw; 12-component conditional interval [−0.000019,+0.000716]). A separate trained-feature/GRU-student proposal already exists as [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17); it has no B300 profile or fit, and its teacher registry remains pending. The gate result does not validate or invalidate that study.

The completed [Kalman output-filter screen](../research/campaign_2026-10-18/online_features_r1/kalman_output_filter_r1/README.md) compared the same raw parent with fixed Q/R=0.25 and first-99-prediction-estimated Q/R on 468 exposed F0 groups/141 components. Scores were 0.6812265, 0.6755207 and 0.6647620 WP respectively; both filters lost in all four descriptive partitions with wholly negative paired component-bootstrap intervals. This rejects these two output-smoothing recipes on this endpoint, not Kalman-derived input/hidden-state features or online learning. Independent review and resource receipt are linked in the packet; no additional B300 draft was warranted, while the distinct PR #17 teacher/student direction remains open.

The [TL3/R3 fixed-mixture F2 replay](../research/campaign_2026-10-18/online_features_r1/tl3_fixed_blend_f2_r1/README.md) completed on 443 previously inspected groups/144 components. TL3 scored 0.723887; R3, fixed 75/25 and fixed 50/50 scored 0.715413, 0.722678 and 0.720950. Their paired conditional component intervals versus TL3 were all negative. The first scorer attempt failed after accessing one target/mask; the correction, amended analysis recipe and successful grade are separately recorded. F2 is exposed and TL3 ancestry is unknown, so this is robustness evidence for those exact endpoints, not clean CV or transfer. It establishes no new large direction; existing B300 handoffs #17/#18 remain open and no new draft was justified by this comparison.

The [early-to-late sequence preference diagnostic](../research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md) completed on 468 exposed F0 groups/141 components. The early label-informed TL3/R3 selector scored 0.693871 late, −0.003798 versus TL3 (conditional 95% component interval [−0.005690,−0.001453]) and −0.000587 versus fixed 75/25 (interval [−0.002360,+0.001586]). Observed early/late expert-choice agreement (68.16%) is below independent-marginal expectations (70.16%/68.64%). Independent source and result reviews pass. Both CPU runs used 2 GB reservations and had zero OOM deltas; the profile materialized the full compressed target/mask arrays while scoring only eight groups. This narrows the frozen label-informed router, not broader online learning. No new B300 draft follows; existing PR #17 remains a separate Kalman feature teacher/student proposal.

The [groupwise fast/slow Kalman state screen](../research/campaign_2026-10-18/online_features_r1/group_fast_slow_r1/README.md) then completed an OOF linear proxy on 32 previously exposed components. Raw+39 group states scored 0.255766 WP versus 0.268859 raw (delta −0.013093; 95% component interval [−0.033693,+0.005956]); the scalar clock was effectively tied (delta −0.000108). The group arm lost in all four inner folds and position bins. Since train lacks the official score mask, this is a nonofficial all-post-warmup train-row proxy, not a validation result. It defers the simple additive ridge encoding only; recurrent Kalman use remains open. Existing [B300 draft #14](https://github.com/VladKochetov007/WunderFunder/pull/14) was updated with this caveat; no separate large direction emerged.

The subsequent [matched recurrent screen](../research/campaign_2026-10-18/online_features_r1/kalman_recurrent_groupfast_slow_r1/RESULTS_R1.md) compared active 39-state inputs against an equal-width zero control on one exposed F0 fold. Scores were 0.67986413 and 0.67981250 WP (Δ +0.00005163; conditional 141-component 95% interval [−0.00013143,+0.00027018]); position bins are mixed. This is a one-seed incremental fine-tune over a common trained parent, not an independent replication or long-horizon online-learning test. The [independent screen review](../research/unified/reviews/GROUPWISE_FAST_SLOW_RECURRENT_SCREEN_REVIEW_20261018_R1.md) found no scoring mismatch and recommends no new B300 draft from this result. Keep the family open; existing PR #17 covers the broader Kalman teacher/student direction.

## Operating state

The latest research closeout is dated 15 October 2026 at 18:39 UTC. It records
that the admitted G fit, export and grade finished, all admitted work drained,
and the research admission barrier remained closed. The latest user explicitly resumed the sequence-moment/heuristic investigation locally on main with a5GB aggregate task RAM cap. This supersedes that pause for this exact scope; unrelated campaign admission and the other agent's Kaggle ownership remain unchanged. Older plans, quotas and stop snapshots are historical authority only.
See the [pause closeout](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md).

The separately recorded October 17 research-reset scope resumes research thinking, literature, EDA and bug repair while excluding model training for that packet. The geometry-specific local fit authority above is separate and does not expand that packet's scope.
The [research reset contract](../research/campaign_2026-10-17/research_reset_r1/CONTRACT.json)
records this boundary. No fit, rental or submission is authorized by this reset.

The [research reset packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
records deployed creative prediction/cache repairs, deterministic checks,
full-data target/event census, sampled full-feature EDA, source registers and
independent proposal/critique rounds. Its ideas are proposals, not new model scores.


## Current scoped geometry investigation

The [geometry record](../research/unified/validation/SEQUENCE_MOMENT_GEOMETRY_20261017_R1.md) contains 128-train/128-valid raw-input EDA, visual inspection of all 112 inputs, and 33 matched tiny-GRU endpoints across two capacities. Independently verified pooled moment gain is +0.0008833 with opposing exposed-fold effects; this is not a production or public gain. The discrete occupancy/dwell and capacity-interaction pilot completed from immutable source snapshots. R3 then trained 12 H128×2 endpoints for eight epochs: all-feature averaged 0.359690 versus 0.376941 for control; dictionary-only averaged 0.377442; a fixed half control/dictionary mixture was +0.007337 over control on the same tiny panel. These are three-seed, 43-fit/46-evaluation-sequence exposed screens, not significance or transfer evidence. The user directed a stronger from-scratch H512×1 feature/control comparison on broader component membership. Its R4 worker was subsequently stopped and drained under the deadline instruction; the previous in-progress status is historical. Earlier startup attempts are preserved and produced no fits. Cloud/team work and the other agent's Kaggle ownership are unchanged.

The [R4 plan](../research/campaign_2026-10-17/geometry_from_scratch_r4/plan.json) specifies 30 paired seed/fold comparisons (60 from-scratch H512×1 fits), fixed epoch 2, equal-width zero-feature controls, component-excluded fitting membership and separate seed/component uncertainty. All four exposed/potentially exposed folds have 443–505 evaluation sequences and 141–149 components each. Full-data training and prediction profiles passed under a 5,000,000,000-byte, one-core, no-swap cap: peak RAM was 2.71 GB, CUDA allocated/reserved peaks were 0.91/1.29 GB. [Admission R3](../research/campaign_2026-10-17/geometry_from_scratch_r4/admission_r3.json) binds verified inputs and current sources; attempt R4 was later stopped/drained for the deadline deliverable. The full matrix is estimated at 113.8 GPU-hours plus setup and exports; no R4 quality score exists.

## Public leaderboard snapshot

The official leaderboard was read on 17 October 2026. The team was at public
WP **0.66832**, rank **22**, with **97 team submissions**; the leader was at **0.67243**. This is a dated
external team result, linked to the [official leaderboard](https://wundernn.io/connectome/leaderboard).
The page does not establish which exact ZIP or submission ID produced the
current team-best score. The result is not private-leaderboard evidence and
does not replace local validation.

The user's later portal screenshot shows **99 total submissions** and **2/5**
used today. It reports R2 at 0.66428 and R3 at 0.66456; the team best remains
the reported TL3 score 0.66832 in the provided results. See their exact
receipts above. This is user-supplied portal evidence, not an independently
queried leaderboard snapshot.

## Latest measured local evidence

The October17 [no-fit packet](../research/campaign_2026-10-17/research_reset_r1/README.md)
adds deployed legacy fixes and a four-group frozen-checkpoint replay (WP0.06289742
with wrong preprocessing,0.72994782restored; bitwise manual/config parity). This
is bug isolation, not a new model ranking. Its full four-column census refutes
H-001 with1,253need-eligible exceptions. The shared evidence and25proposal branches
are published on research-team; [publication status](../research/campaign_2026-10-17/research_reset_r1/publication/README.md)
records that new PRs/main promotion await API authentication.

The last completed comparison in the current pause record is the exposed-F0
G-versus-C grade: G scored 0.68125607, a delta of **+0.00002956** over C, with
a conditional component 95% interval of **[−0.00038018, +0.00037707]**. This
single-seed comparison resolves no quality advantage. See the
[pause receipt](../research/unified/operations/RESEARCH_PAUSE_20261015_R1.md)
and [campaign frontier](../research/campaign_2026-10-13/strategy_execution_r1/FRONTIER.md).

Other late Main results remain scoped to their recorded panels. For example,
the matched short architecture screen measured H512×1 at 0.67323776 versus
H256×2 at 0.66977764 on one exposed F0 seed. That is a conditional screen, not
a converged architecture ranking or public-score forecast; see the
[student frontier](../research/campaign_2026-10-14/cpu_student_frontier_r1/FRONTIER.md).

## Code and draft status

The modular trainer is integrated in Main. [scripts/train_rnn.py](../scripts/train_rnn.py)
is the compatibility command-line entrypoint; reusable training components
live under [src/wnn/training/](../src/wnn/training/README.md). The refactor
receipt reports static structure/lint checks and CLI help validation. It does
not report unit or integration tests, numerical training parity, a new fit, or
a throughput benchmark. Read the
[refactor receipt](../research/unified/methods/MODULAR_RNN_TRAINER_REFACTOR_20261017_R1.md)
before interpreting the change as evidence about model quality or performance.

The subsequent [repository maintenance](../research/unified/operations/MAIN_MAINTENANCE_20261017.md)
adds synthetic CPU regression and integration checks, legacy class compatibility
fixes, and fewer intermediate feature concatenations. This is software evidence;
it establishes no new model score, GPU parity or measured training speedup.

The 17 October GitHub snapshot recorded 14 experiment-prefixed draft pull
requests. They contain implementation proposals and handoffs; the count is not
a count of completed fits. Historical results already present in Main must be
tied to their own receipts. See the
[repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
[pull-request list](https://github.com/VladKochetov007/WunderFunder/pulls), and
[experiment draft workflow](../research/unified/methods/EXPERIMENT_DRAFT_WORKFLOW_20261017_R1.md).

## Navigation

- [Research objective](research-objective.md) — goal and binding constraints.
- [Research library](../RESEARCH.md) — how to find ideas, results and source
  evidence.
- [Repository audit](../research/unified/reviews/REPOSITORY_AUDIT_20261017.md),
  [model catalog](../research/unified/models/CURRENT_MODEL_CATALOG.md),
  [validation protocols](../research/unified/validation/VALIDATION_PROTOCOLS.md),
  and [storage map](../research/unified/operations/STORAGE_MAP.md) — current
  maintained indexes for the 17 October audit.
- [Decision index](adr/README.md) — current decisions and their successors.
- [Complete dated handoff](../research/RESEARCH_COMPLETE_HANDOFF_20261012.md)
  — October 12 archive, not a current state.
- [Preserved state history](current-state-history-20261017.md) — byte-exact copy
  of the previous 199,958-byte file, SHA-256
  37be5efacd617f8e712c99822c0399242949aef10fbeebc9abd20ee90f499902; its
  same-directory location preserves the original relative link bases.
