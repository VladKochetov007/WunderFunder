---
kind: review
status: complete
created_utc: 2026-10-15T15:19:09Z
updated_utc: 2026-10-15T15:19:09Z
hypothesis_ids: [H-STUDENT-PARENT-GROWTH-20261015, H-087, H-138]
experiment_ids: [E-WARM-MECHANISM-WAVE-20261015-R3]
evidence_paths:
  - research/campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_WAVE_R3.md
  - research/campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_COMMANDS_R3.json
  - research/campaign_2026-10-15/gru_parent_growth_r1/TRAINER_SOURCE_MANIFEST_R3.json
---

# Warm-parent growth and age wave R3 recipe review

## Question or claim

Do the frozen R3 worker vectors implement a matched two-epoch F0 seed-3
continuation screen for a shared control, group-age features, per-field-age
features, and a function-preserving wider initializer? This review checks source
and recipe consistency only. It does not admit a fit or establish quality.

## Findings

**Bounded GO for the four 50-update B16 resource profiles, after the controller's
normal parser/source preflight and fresh resource admission. Full fits remain on
hold until root records the terminal profile evidence and separately admits the
fit sequence.** No profile or fit was launched in this review.

The commands bind the clean immutable worktree `56d900e3c283512e7128243374cd937c05979359`
through `TRAINER_SOURCE_MANIFEST_R3.json` (SHA-256
`5949c1cbd9fbef6383efd793aadaf29dc9fc53fd0c7df70058422da65b33b9a7`); the
command packet SHA-256 is
`16b8ae1a843b36b779219fb44ec57fc16b5a978dbc52e35703da9af2e2874a6a` at review
time. The packet uses the actor worktree's absolute `scripts/train_rnn.py` and
`PYTHONPATH`, while `WNN_PROJECT_ROOT` points to MAIN for data and run artifacts.
I independently matched the manifest entries for the trainer, `research_features.py`,
`training_sampling.py`, and `gru_growth.py` to their worktree bytes. The commit
was clean at inspection. This avoids the historical wrapper selecting a stale
trainer; the controller still has to use the specified immutable source path.

The four fits share the same parent ancestry, cache, propensity, split, target
weights and loader geometry. C/G/F load `c20261003_combo_s3_f0@ep3`; W loads
`gru_growth_combo_s3_f0_h384_r1@ep3`. The inherited Alpha0 P1 contract pins the
original parent, clean-T3 cache, propensity and F0 split. The growth README pins
the derived initializer checkpoint at SHA-256
`622a4afc1423fa9fdb976f73dd786e29c946f4710e72c373a69385f1220f1ce0` and records
that it carries the parent's epoch-3 state with zero new training updates and
no optimizer state. Before dispatch, root should verify that exact file and its
parent provenance at the named MAIN artifact path.

The age extension is compatible with `--inherit-init-preprocessing`: the parent
mode is `none`, and `can_extend_research_features` admits only a causal feature
name prefix. `group_age` and `field_age` each append 112 coordinates;
`expand_research_feature_state` zeroes the added input columns and supplies
mean-zero/std-one normalization for them. C's `none` mode is unchanged. W
depends on the derived H384 initializer: the trainer's inheritance code does
not grow hidden width itself, but the derived checkpoint and explicit H384
argument agree. The complete-sequence initializer check is engineering
evidence; it does not establish post-update quality or FP16 serving parity.

The fixed checkpoint and update budget are internally consistent. With
`--epochs 2`, fixed zero-based checkpoint epoch 1 and `--stop-after-epoch 1`,
the trainer runs epochs 0 and 1; its loop stops only when `epoch > 1`. The
campaign split supplies 10,535 train-fit plus 1,405 fit-valid groups, with the
468-group evaluation fold excluded from fitting. `valid_fold=0` forces the
evaluation source to that fold, so `valid_seqs=600` caps a 468-group panel.
The concatenated fit set has 11,940 groups: 746 complete B16 batches per epoch,
7,460 attempted C2000 chunk updates per epoch and 14,920 across two epochs.
FP16 skips may reduce completed updates and must be reported. A separate
post-fit export must still cover all 1,873 valid groups before grading the 468
evaluation members.

The base dataset is ordered by the seed-3 NumPy train-fit permutation followed
by fit-valid membership. The trainer then applies `EpochOrderSampler` to the
complete concatenation: `torch.randperm` uses `seed + 1009*(epoch+1)`, with
epoch -1 for the shared normalization pass and epochs 0/1 for fitting. Thus
samples cross the train/valid boundary in each epoch; all four arms still share
the same source-qualified membership and deterministic order. The R3 command
packet was corrected to state this accurately. The 50-update profile command
uses five B16 batches × ten chunks per arm, 300-second profile timeout, and is
resource/rate evidence only; fit workers retain a 5,400-second cap.

## Recipe differences and limits

| Comparison | Intentional difference | Required match | Remaining limit |
|---|---|---|---|
| C vs G/F | Append group-age or per-field-age features with zero input weights | H256×2, same seed, parent, membership/order, objective, alpha .9, two-epoch OneCycle and CP1 | Feature rank and optimizer geometry change; this is not a clock-only contrast |
| C vs W | Use a checked H384×2 function-preserving grown initializer | Parent ancestry, data/cache, seeded sampler, optimizer, updates and output targets | Capacity and optimization geometry change; derived checkpoint hash must be checked before dispatch |
| All arms | Continue the prior distilled CP3 parent with clean-T3 alpha .9 | Clipped weighted label objective, teacher cache, propensity, auxiliary weights, batch/chunk and fixed checkpoint | Exposed F0, one seed, short horizon; not cold-start KD or a new clean holdout |

The vectors remain marked `fit_admitted: false` and
`SOURCE_BOUND_PREPARED_REQUIRES_REVIEW_PARSER_AND_FRESH_ADMISSION`. Required
next evidence is the exact source-bound parser/preflight result, confirmation
of parent/cache/propensity/split and derived-initializer hashes, fresh B16
resource admission, and terminal per-arm profile receipts. Those checks should
be done by the root controller; this review adds no approval gate. Profile
success does not itself admit or validate a quality fit.

## Checks and limits

Read the R3 wave and JSON vectors, the pinned trainer/research-feature/sampler
sources, the reference Alpha0 P1 contract and source manifest. Verified source
hashes for four relevant files, worktree HEAD and clean status. Read the
trainer's init inheritance/feature expansion, fold membership selection,
concat loader, sampler, fixed-checkpoint and stop-after-epoch logic. No Torch,
checkpoint, array, GPU or fit was used. The root's complete-sequence profile
and nine focused tests are prior engineering evidence only; this is not an
independent rerun.

The post-fit prediction/grade commands are not part of this worker packet. Any
result must remain an exposed single-seed F0 discovery screen until evaluated
with the unchanged official metric and later checked across source-qualified
folds and fitting seeds.

## Source records

- [R3 mechanism wave](../../campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_WAVE_R3.md)
- [R3 worker vectors](../../campaign_2026-10-15/gru_parent_growth_r1/WARM_MECHANISM_COMMANDS_R3.json)
- [Trainer source manifest](../../campaign_2026-10-15/gru_parent_growth_r1/TRAINER_SOURCE_MANIFEST_R3.json)
- [Growth engineering receipt and derived initializer hash](../../campaign_2026-10-15/gru_parent_growth_r1/README.md)
- [Warm-parent Alpha0 reference contract](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_P1_FIT_CONTRACT_R2.json)
