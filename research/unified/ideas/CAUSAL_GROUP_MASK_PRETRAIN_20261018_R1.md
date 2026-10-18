---
kind: hypothesis
status: proposed
created_utc: 2026-10-18T06:00:51Z
hypothesis_ids: [H-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
experiment_ids: [E-CAUSAL-GROUP-MASK-PRETRAIN-20261018-R1]
parents:
  - family:teacher
  - family:pretraining
evidence_paths:
  - ../sources/CAUSAL_GROUP_MASK_PRETRAIN_SOURCES_20261018_R1.md
  - ../../experiment_drafts/causal-group-mask-pretraining-b300/README.md
---

# Causal group-masked pretraining for the recurrent teacher

## Hypothesis

A GRU teacher pretrained to reconstruct temporarily hidden raw feature groups from the other visible groups and its causal past may learn a reusable cross-channel state. The user-provided raw-column schema supplies 13 meaningful masking units. This may improve the downstream teacher after full-input supervised fine-tuning and potentially give a better teacher for a later GRU student.

## Why this is distinct

The active B300 drafts already cover next-row self-supervision (#18) and per-sequence online parameter updates (#20). This proposal hides one current raw feature group and reconstructs it from the same row's remaining channels plus past state. It uses no future context, labels or test-time weight updates. The pretraining is feature-only and fold-clean. A matched 8-epoch supervised control checks whether the same compute is better spent directly on labels.

## Predicted observation and falsifier

If the representation hypothesis is useful, the 2-epoch masked initializer followed by six supervised epochs should outperform both a cold six-epoch fit and a cold eight-epoch compute-matched fit on the fixed official outer-fold metric. Lower masked reconstruction loss alone is not evidence of task value. A clear negative result narrows this group-mask schedule, GRU width, pretraining budget and transfer path; it does not reject next-input prediction, temporal patch objectives, all masked modeling or online learning.

## Data and causal contract

For each cell, statistics, pretraining, initialization and supervised fitting use only that fold's source-qualified fitting components. The loader omits target and scoring-mask columns. Event features are rebuilt from the masked raw inputs before entering the pretrainer. The recurrent state resets for each sequence, advances in order and is detached only at the declared 2,000-row TBPTT boundaries. Fine-tuning and serving use complete current/past inputs and deterministic recurrent state reset. Existing folds remain exposed; no fresh-holdout claim is made.

## Evidence and allocation

The [primary-source review](../sources/CAUSAL_GROUP_MASK_PRETRAIN_SOURCES_20261018_R1.md) provides external motivation and failure modes; none of its benchmark scores transfer numerically. The [B300 packet](../../experiment_drafts/causal-group-mask-pretraining-b300/PR.md) includes runnable code, a strict-FP32 profile gate and 30 paired seed/fold cells for each of three arms. No local fit, B300 run, LB impact, or expected numerical gain is claimed. The main follow-up, if teacher value is supported, is fold-clean student distillation and a separate served endpoint comparison.
