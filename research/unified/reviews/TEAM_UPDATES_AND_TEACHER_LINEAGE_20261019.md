---
kind: review
status: complete
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - specs/current-state.md
  - research/campaign_2026-10-19/full_gru_tweaks_r1/
  - research/unified/reviews/CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md
  - research/unified/models/CURRENT_MODEL_CATALOG.md
  - research/unified/ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md
---

# Team updates, teacher lineage, and Transformer status

## Audit boundary

This audit compares `17f0bd6b3005f2631a96a362fc4cf17fd1ef4f33` with `origin/research-team` at `3cebc9f0a0e375c71651115ca43bad129af857e8` (32 commits ahead; parent `87e89de1630cc62bc7a891d5029582b85314c0e5`). The added commit records accepted strong-GRU prerequisites, a passing full-shape profile, and the first actual clean teacher launch. Evidence below is pinned to that ref and the dated receipts it contains.

The companion Transformer memo and three experiment packets are additions outside this audited ref. They are design material only and are not evidence of an implementation, fit, profile, or result. No job or provider state was queried or changed for this review.

## Current result updates

- The latest full-GRU continuation says the one-GPU study is authorized for five arms: strong control, EMA, sparse residual factors, mean/log-variance objective, and all three. Each arm and its clean ancestry remain separate; the full 30-pair-per-arm matrix is a plan, not an allocation or a result.
- New F0 exploratory candidate: sparse+EMA with NLL scores pooled WP `0.5044141`, compared with stored sparse-WMSE `0.4824412` and dense-WMSE `0.4798731`. It is one adaptive exposed-panel/seed comparison. The missing dense+NLL arm prevents a synergy estimate; this is not a GRU, public, full-fold, or CP30 result.
- R15 sparse-vs-dense scored `0.4824412` vs `0.4798731` (`+0.0025681`), below the predeclared `+0.005` allocation gate. It is a retained small candidate, not a promoted result or evidence of stable factors/equal compute.
- The older F3 cold labels-only GRU pair scored wide H512×1 `0.7031225` and deep H384×2 `0.7032378`; deep-minus-wide is `+0.0001154`, conditional component interval `[-0.0024723, +0.0020551]`. This is one seed on one exposed F3 fold (3/30 cells per arm in the wider plan), not a width/depth winner, teacher, or Transformer result.

These screens and F3 do not measure the new teacher or any proposed Transformer transfer path. CPU-screen deltas cannot be added to estimate a full-GRU or public improvement.

## Strongest source-supported fold lineage

The existing historical Main T3 F0 lineage is the strongest measured clean-fold teacher/student candidate in the reviewed recurrent evidence, but F0 is repeatedly exposed and its score is not a fresh-holdout claim. The three fixed-epoch-2 teachers scored `0.67898918`, `0.67836691`, and `0.67969849`; their equal-prediction trio scored `0.68090199` on 468 groups / 141 components / 808,863 scored rows. That historical split is `09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d`, distinct from the current F0 source-qualified split.

The historical clean T3 seed-3 student scored `0.68115536` vs its warm parent `0.68094121`; the paired 95% component interval is `[-0.00338452, +0.00309511]`. The child inherits an earlier distilled H256×2 parent and changes target, optimizer, precision, batch/chunk geometry, and update budget. Parent inference precision is unrecorded. This does not isolate the causal benefit of the new teacher target or show equal teacher/student ceilings.

The current study is an explicit clean rebuild adaptation, not exact historical reproduction: strict FP32 and current-source FP32 Muon-all semantics, current F0 membership, and a newly fitted propensity replace incomplete/older ancestry. The current split is `c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7`; do not transplant historical membership counts or call its scores current.

The public reference is a different endpoint. The audited team snapshot records `kotiki` at `.66832`, rank 22/97 submissions; `tl3.zip` was recorded at `.6683 / 39:12`, 12,538,394 bytes, as a three-member recurrent blend. The local ZIP hash is not bound to the uploaded portal payload, and the exact portal ID-to-ZIP mapping is not verified. Members trained on final train+valid; their own validation scores are fitted diagnostics. This public recipe is not a clean outer-fold teacher and its public metric cannot be compared numerically with local F0 WP.

## New clean teacher: launched, not scored

The accepted prerequisites are real terminal artifacts: clean F0 propensity completed four epochs / 3,520 updates / zero nonfinite updates, with model SHA `6facf4ca9045118b8e18ca590053c97e2f5f5122c92f228f75002f0c76a2800e`; the shared train-only dictionary used 64 fitting sequences / 31,808 rows / 756 updates. The dictionary artifact SHA is `9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659`.

The current-source H384×2 teacher profile passed 120 finite strict-FP32 Muon updates on the actual training shape. It produced no model checkpoint and no quality grade. A single-worker GPU reallocation moved the authorized worker to an available identical A4000 while preserving foreign jobs; earlier resource failures remain in the record.

Teacher seed 3 then launched at 15:35:05 UTC, invocation `02a8c00b433b49a185c2700624397769`, source `af21d6d8409c46b36830f0bdddd50e67ee17025f` (164 files). It is H384×2, 23 outputs, batch 32 / chunk 1,000, three epochs with fixed epoch 2, strict FP32 and the profiled Muon-all port. Fit membership is current F0 train+valid with held-out validation excluded. The configured 22,380 updates are expected work, not completed updates.

The latest saved startup observation is 15:40:38 UTC: the exact invocation was active and its log showed epoch 0, batch 20/373. No checkpoint, terminal receipt, teacher score, shared parent, or five-arm student result is recorded at this snapshot. The capture note says its raw files match the pinned schema, but the local wrapper failed before its own persistence and the raw-file origin / SSH return code remain unknown. No retry was made; this uncertainty does not turn the startup observation into terminal evidence.

The frozen rebuild plan calls for teacher seeds 3/4/5, then an independently fitted H256×2 seed-1001 parent and five common-parent student arms. That is future recipe scope, not launched work: `3cebc9f` proves only seed 3 startup. A single fit cannot be described as the planned teacher trio or used as evidence for the parent/children before its own terminal checkpoint and provenance acceptance.

The teacher source is published at `af21d6d8`; its all-scope FP32 Muon behavior is an explicit port. Historical Main T3 evidence used incomplete or different precision/runtime ancestry, so the new fit tests a source-bound adaptation. Even a completed teacher grade would remain an F0 teacher result, not a measured distillation win or public forecast.

## PR25–27 and execution readiness

PRs [#25](https://github.com/VladKochetov007/WunderFunder/pull/25), [#26](https://github.com/VladKochetov007/WunderFunder/pull/26), and [#27](https://github.com/VladKochetov007/WunderFunder/pull/27) remain open drafts at heads `967f2f342e919e69f4f6cee39373b3b01b25c16a`, `e713efc5aa67b82f36826105df5c4232c7fc312b`, and `14281c47e1b3d77ec5bb2f86d0cb460b11f66b9f`. They are cold labels-only GRU comparisons, not teacher or Transformer fits.

PR25's older 14:30 status comment (`0/60`) is superseded by its user-requested stop/recovery comment. That comment records the owned B300 work stopped and verified at 14:33:55 UTC, with zero active tabdata units, project processes, or project CUDA clients. Without refitting or regrading, 12/30 existing paired official grade cells (24 scored fits) were receipt-verified: equal-fold wide `0.68409111`, deep `0.68331379`, deep-minus-wide `-0.00077732`. This is partial exposed-fold evidence, not a complete matrix, significance claim, or public forecast.

PR26 (field age) and PR27 (causal innovations) last reported `0/30` matched grades and four active paired cells before the same stop. They remain unscored partial comparisons; the earlier active status is not current readiness to resume. Preserve artifacts and stop boundary.

The same [stop and score-recovery receipt](https://github.com/VladKochetov007/WunderFunder/pull/25#issuecomment-6268923635) preserves geometry PR15:64/120 scored fits,16/30cells per arm; equal-fold baseline0.67824425, moments0.67809108, dictionary0.67756978, combined0.67702747. These partial differences are all negative and do not establish significance or invalidate distinct descendants. Geometry PR15 is different from the CPU screen named R15 above. [Pinned stop proof and preserved scores](https://github.com/VladKochetov007/WunderFunder/blob/967f2f342e919e69f4f6cee39373b3b01b25c16a/research/campaign_2026-10-18/b300_execution_r1/stop_20261019_r1/README.md) retain exact scope.

## Transformer and distillation: proposal, not implementation

The audited Transformer memo remains `status: proposed`. It proposes a causal numerical-input teacher with a two-scalar output autoregression outside the attention backbone, optional future-observable-change supervision as a separate objective, FP32 reference training, cached teacher predictions, and downstream GRU distillation. Its eight-B300 capacity, throughput, software, and fit assumptions are user-reported/planned; there is no verified cluster profile, run, Transformer score, cache, or distilled student result.

The tracked model surface is still recurrent. In `src/wnn/nn.py`, `RNNModel` uses a GRU for time recurrence; `enc="tokens"` applies a small Transformer over feature-group tokens within each time row before the GRU. `src/wnn/block_memory_attention.py` is a separate block-memory architecture exposed by the training model factory, not the proposed full causal Transformer teacher/readout and transfer pipeline. `src/wnn/distillation.py` supports cached teacher-family targets and student losses but does not implement the proposed Transformer training/export path. `src/wnn/infer.py` exports and steps `NumpyRNN`; it has no temporal Transformer stream/cache implementation.

PR24 remains an open draft, head `954f789b3c734d24468fb5cbba7ddfe7484cb8d0`. Its body describes a legacy trainer at `bd78ac1048fe9d6534e76cde8c6f8ec0d6aa6acb`, only an unrun F0/seed-3/one-epoch screen, an unbound current-source run owner, `launch_ready=false`, `training_authorized=false`, and no runnable current-base command. It is separate from the newly launched clean GRU teacher.

The companion deliverable adds transfer-frontier, representation-hint, streaming-student, and CPU-inference drafts. Those are planning documents outside audit ref `3cebc9f`; their more detailed recipes do not change the implementation or readiness status above.

## Evidence navigation

- [Full GRU continuation](../../campaign_2026-10-19/full_gru_tweaks_r1/README.md), [teacher startup](../../campaign_2026-10-19/full_gru_tweaks_r1/records/teacher_s3_v1/README.md), [teacher profile](../../campaign_2026-10-19/full_gru_tweaks_r1/records/teacher_profile_v3/README.md), [propensity terminal](../../campaign_2026-10-19/full_gru_tweaks_r1/records/propensity_f0_e4_v3/TERMINAL_ACCEPTANCE.json), [dictionary acceptance](../../campaign_2026-10-19/full_gru_tweaks_r1/records/dictionary_f0_v1/ACCEPTANCE.json).
- [Sparse+EMA/NLL result](../../campaign_2026-10-19/idea_combo_r1/RESULTS.md), [R15 result](../../campaign_2026-10-19/idea_screens_r15/RESULTS.md), [historical clean T3 audit](CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md), [model catalog](../models/CURRENT_MODEL_CATALOG.md).
- [Original Transformer design and new transfer comparisons](../ideas/CAUSAL_AR_TRANSFORMER_GRU_DISTILLATION_B300_20261017_R1.md), [CPU inference contract](../inference/TRANSFORMER_DISTILLATION_CPU_PATH.md), [architecture handoff PR24](https://github.com/VladKochetov007/WunderFunder/pull/24).

## Source pins

- Snapshot: `specs/current-state.md` `ff73129b21fc5345816a83edb33b2617c87851ab`; full-GRU overview `9703452182f44823a482ba0e56693066d9a2dd34`.
- Current teacher evidence: teacher run README `bacdaf12a68a513ef6a00372a59776fd08038210`; startup observation `8ec61a1c48bc1b8e0468a325723ff1a09436e77f`; profile README `95305944689ac929020d0d7f356f7e353a8f2c32`; propensity terminal acceptance `36b7bbdb64e207531759345aad4d92ec114ebb73`; dictionary acceptance `84599972e9c9c87fc8e1de477acc001930efaea8`.
- Results: R15 `a546d3c54417c60573f826d4b7db4bf1f59e27d8`; sparse+EMA/NLL `389e4062a1f3d1be13ed582505a452a42e34f210`; F3 collection `6f345e7679c7335efd88587aea6b933ee2b3c600`; clean T3 lineage review `c787090f62fc4d8f52696aa68991b65038f83fb7`; public model catalog `23df37e7f9c739b7177b02ec77f3b8f4eda005a0`.
- Transformer source pins: proposal `fd0c48f7626f297675120f174a2fededb90f521c`; `nn.py` `ae70ecf66555a3203bef501ec8abe9cd9bcef8f5`; `block_memory_attention.py` `a529f3a04929a99bb1984e97a6adf01dd2a1efb4`; training models `2794dad5716d4c27b80b2afc8ed843da102a7548`; distillation `1b6e3cab22f1cd2665bb59aac0eb4d721da79f62`; inference `84416b4d8c01556c312a572ac1842925d7782a61`.
