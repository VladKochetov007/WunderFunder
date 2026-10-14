---
kind: review
status: complete
created_utc: 2026-10-14T01:31:33Z
updated_utc: 2026-10-14T01:31:33Z
hypothesis_ids: [H-STUDENT-LADDER-20261014]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_SEED3_B8_R2_CONSUMER_CONTRACT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/CPU_PREFLIGHT_R2_RESULT.json
  - research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/TRAINER_CLI_ATTEMPT_R2.json
---

# CPU student-frontier consumer and CLI profile review R2

## Question or claim

Does the frozen R2 B8 consumer bind the actual clean-F0 producer, fitting membership, common normalization and production sampler order closely enough to admit its one-batch trainer CLI profile?

## Mechanism and method

Reviewed the sealed source at commit `f0b5b6eb469d960708b590e7dd4814e3bd2daa19`. The trainer, consumer and contract hashes are `952939f15e1dfa84bdee20295a4c64c1fbe4fc174d72e76a6bf574d2fdaeba01`, `a747431d9bc2acc25f3f053bdfe7b5cfd64c4a211ecdd75002faa54785062e07`, and `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`. The consumer leaves the producer's historical cache validator and metadata unchanged, then binds the separate one-epoch F0 seed-3 B8 recipe, exact source-qualified split and sampler, teacher and propensity ancestry, payload hashes, and H256×2 mean/std buffers. It does not load learned normalization-source weights. The reviewed order fix maps concatenated dataset positions by the train/valid dataset boundary, so row-group 37 in one parquet cannot be confused with row-group 37 in the other.

## Evidence and interpretation

Five focused sampler/source tests passed under a one-CPU, 1 GB task cap. The producer-backed CPU no-fit receipt pins metadata SHA256 `1e4c5c4915b643352494558efc17996a11900224231405fa4ab05698f2be3dc8`, verifies both target payloads, reports 11,940 fitting groups and 14,920 planned updates, and records `optimizer_constructed=false`, `fit_started=false`, and no quality claim. The subsequent actual trainer CLI smoke completed ten FP32 chunk updates with finite losses and gradient statistics and zero skipped updates. Its host peak was 2,464,940,032 bytes; the controller's sampled VRAM peak was only 269 MB and is too sparse to replace the all-shape profile's stronger allocator/VRAM evidence.

This clears the one-batch H256×2 CLI/resource check. It does not grade the model, rank the eight shapes, or admit seven additional fits. Root's separate current allocation admits the H256×2 full fit only after its fresh resource gate.

## Limits and validity

F0 is an exposed development split. The no-fit receipt proves strict consumer preflight and the CLI smoke exercises one B8 batch (ten C2000 updates); neither establishes full-epoch quality, robustness across seeds/folds, or submission performance. The prior B1 reader's overlapping-group sampler-source claim remains withdrawn in its immutable correction receipt; this R2 path uses the actual trainer's separate train/valid datasets and the corrected source-qualified order digest.

## Decision and next step

Bounded GO for the actual trainer CLI profile only, then for the root-admitted H256×2 control fit under its independent live memory/time admission. Keep the remaining ladder shapes unadmitted until the control's progress and observed resource use are reviewed. Preserve full checkpoint, update-counter and provenance evidence before prediction export. No quality promotion follows from this review.

## Source records

- [Frozen matched B8 recipe and exact commands](../../campaign_2026-10-14/cpu_student_frontier_r1/recipes/README.md)
- [R2 producer preflight and order receipt](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/CPU_PREFLIGHT_R2_RESULT.json)
- [R2 actual trainer CLI attempt](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/TRAINER_CLI_ATTEMPT_R2.json)
- [All-shape B8 resource profile](../../campaign_2026-10-14/cpu_student_frontier_r1/profiles/PROFILE_B8_LADDER_R1_RESULT.json)
- [Research hypothesis](../../hypotheses.jsonl) (`H-STUDENT-LADDER-20261014`)
