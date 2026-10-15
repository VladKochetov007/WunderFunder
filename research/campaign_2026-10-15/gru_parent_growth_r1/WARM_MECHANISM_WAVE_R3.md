---
kind: experiment
status: proposed
created_utc: 2026-10-15T15:15:19.360192+00:00
updated_utc: 2026-10-15T15:15:19.360192+00:00
hypothesis_ids: [H-STUDENT-PARENT-GROWTH-20261015, H-087, H-138]
experiment_ids: [E-WARM-MECHANISM-WAVE-20261015-R3]
evidence_paths: [COMPLETE_SEQUENCE_PROFILE_R1.json]
---

# Shared-control warm mechanism wave R3

## Question or claim

Does more inherited trainable capacity or finer causal change-age information
improve a strong common parent at a fixed continuation budget? Root owns profile,
fit, complete predictions and official paired grades. This is a proposed bounded
allocation after the healthy six-epoch alpha0 chain, not simultaneous GPU admission.

## Mechanism and method

| Arm | Width/depth | Features | Initializer |
|---|---|---|---|
| C | H256×2 | original211 | original parent |
| G | H256×2 | original211 +112 broadcast group ages | original parent, zero appended columns |
| F | H256×2 | original211 +112 per-field ages | original parent, zero appended columns |
| W | H384×2 | original211 | checked function-preserving grown parent |

All inherit `c20261003_combo_s3_f0@ep3`, including its prior distillation ancestry.
Preserve normalization, all23 output coordinates and all original features. G/F
append mean0/std1; W uses the derived initializer indexed in this directory.
Use common seed3, **`--batch-order seeded`**, B16/C2000/FP16, fresh AdamW at
LR1e-4/WD.01, OneCycle horizon **two epochs**, fixed CP1, auxiliary weights.3/.3,
homogeneous clean-T3 cache, alpha.9, and unchanged clipping/weighting/propensity.
The exact inherited optimizer/objective settings are those in the existing
alpha0 continuation contract except for the explicitly changed alpha, horizon,
checkpoint, sampler and arm-specific features/initializer/width. Freeze runnable
commands and source hashes before profiles; this narrative is not a launch command.

Use source-qualified F0 membership:10,535 train-fit +1,405 valid-fit;468 valid
evaluation groups excluded from fitting and ancestry.746 drop-last batches per
epoch give7,460 attempted chunk updates,14,920 total. Record completed updates and
AMP skips. Explicit seeded order avoids architecture-dependent global-RNG shuffles;
train-fit order first follows the recipe's seed3 NumPy permutation, then fit-valid.

Export all1,873 valid groups in canonical order, then grade468 evaluation groups
with unchanged `global_wp_clip2_v1`. Primary contrasts F−G and W−C; F−C/G−C are
secondary. Report paired2,000-component bootstrap intervals and position bins,
with one-seed, exposed-F0 and short-horizon limitations. Shared C saves one entire
control fit relative to two separate experiments. Do not reuse historical alpha.9
CP1: its six-epoch schedule/global sampler does not match this new control.

## Evidence and interpretation

Early matched continuation favored alpha.9 over alpha0 by.00394686 after two
epochs of a six-epoch schedule. This motivates preserving KD here provisionally;
it does not settle the six-epoch comparison or every KD interaction. The current
alpha0 experiment and final checkpoint rule remain unchanged.
The growth initializer passes complete-sequence FP32 engineering checks; age
production-Torch/oracle tests pass. Neither is quality evidence.

## Limits and validity

This identified R3 supersedes the **unlaunched** age R2 alpha0 allocation; R1/R2
remain archived. F−G changes feature rank/optimizer geometry as well as clock
resolution. Growth adds capacity and changes optimization geometry. New width
and feature modes require actual trainer reload, initial prediction parity,
representative new-parameter gradients and actual B16 FP16 memory/VRAM profiles
before fits. No native runtime, quantized parity or submission readiness follows.

## Decision and next step

Freeze four source-bound commands and independently review the exact consumer.
Profile serially after the active chain; fresh admission must meet two aggregate
CPU cores,17GB committed target,18/20GB memory limits, no swap, one local GPU
worker and5GB observed VRAM. Scratch stays on physical disk. Root must fit,
export and grade admitted arms rather than stop at code. Promote useful candidates
to longer matched horizons/seeds/folds; a short negative does not kill a family.

## Source records

[Growth and test receipts](README.md),
[unlaunched age R2](../vast_a_cpu_probe_r1/per_field_age_r1/MATCHED_SCREEN_R2.md),
[KD comparison](../student_quality_tradeoff_r1/README.md).
