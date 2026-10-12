> Extracted from the final October 12 handoff. Its original SHA-256 was `e033f8d52f4d4444e5f0c5fa006c2b4f56b6e82f7187e2ed9fb1b74b20e6f704`. See [archive index](../README.md) and [split manifest](../SPLIT_MANIFEST.json). Local links were rebased for this page.

## Model families, training, and source-qualified findings

The core family is recurrent GRU sequence modeling with independently trained
teachers and students. Training and distillation use supplied allowed data;
the all-data final is distinct from a held-out evaluation sibling. Teacher
targets use the declared clipped mixture
(1 − alpha) * clip(y, −2, 2) + alpha * clip(teacher, −2, 2) where specified,
while primary MSE continues to use clipped true-label magnitude weights and
the existing auxiliary objectives. Alpha 0.9 has a favorable result in the
tested one-seed ladder; it is an exploitation setting, not a universally
established optimum. The main-target alpha=1.0 arm did run on exposed F1: 0.67068 versus alpha0.9 0.67103 and alpha0.7 0.66991 (six epochs, same cached targets). That is teacher-only primary-target mixing, but true-label magnitude weights and auxiliary supervision remain; a strict entirely label-free objective was not run. See [deep dive](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/DEEP_DIVE_CURRENT_BEST_20261011.md) and LOOP_LOG passes125/128. These one-seed results do not establish a universal optimum.

The clean-T3 CP5 controls are the most recent completed local student grades. The independent review reports 0.6811553603333363 for seed 3,
fixed epoch 5, on 468 exposed F0 groups / 141 components. The source-qualified
fit used 10,535 train groups plus 1,405 valid fit groups, excluding 468 F0
evaluation groups. It starts from own checkpoint
c20261003_combo_s3_f0@ep3; its three own H384 Muon teachers are seeds 3/4/5
at epoch 2. The full-valid FP32 prediction export is finite and hash-bound.
The actual homogeneous cache target is Tstored = float32(0.8125 *
float32_mean(three teacher predictions)), without cache clipping. Training
then uses 0.1*clip(y,−2,2) + 0.9*clip(Tstored,−2,2). Therefore the nominal
pre-clipping contribution is 0.24375 per teacher; provenance rows labelled
0.1 are source-parent metadata, not the numeric mixture weights. Clipping
makes the final operation nonlinear.

The admitted seed4 control is now complete at WP **0.6812584217** (t0 0.6742837474, t1 0.6882330961) on the identical exposed468-group F0 panel. Seed3/4 mean **0.6812068910**, sampleSD **0.0000728754**, n=2 student/seed-matched-initializer fits sharing one teacher cohort. This is not cross-fold confirmation, independent teacher replication, or a measured student ensemble. No public forecast follows. Six epochs used44760attempts/44740completedupdates/20AMPskips; actualCP5 `e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c`, complete finite FP32 prediction `f0689f9596d5319e7f5f2e54c5598168f17bcfd926e93e7724fc3878103edec8`. See [seed4 final handoff](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/FINAL_HANDOFF_R1.md) and [machine receipt](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/SEED4_CONTROL_FINAL_RECEIPT_R1.json). The generic teacher coefficient rows and shorthand target formula in raw receipts are lineage/nominal metadata: the actual clipping and .8125 mean scale remain as explicitly described above.

The closest old Z4 student score is 0.6831418134, from a different
teacher/initializer lineage, recipe and FP16 forward inference, versus clean
T3 FP32. This endpoint gap cannot isolate target family, precision, or
architecture. The full static review, hashes, terminal, teacher cache, and
grade links are in
[the CP5 grade review](../../../../campaign_2026-10-12/clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md)
and its JSON receipt.

Clean T3 teacher evidence on the same exposed F0 panel:

| Endpoint | F0 WP | Scope |
|---|---:|---|
| H384 Muon teachers, seeds 3/4/5 | 0.67898918 / 0.67836691 / 0.67969849 | three fit seeds, same exposed F0 panel |
| Teacher-seed mean | 0.67901819 | sample SD 0.00066626 |
| Fixed equal teacher trio | 0.68090199 | +0.00188379 vs seed mean |
| Old student + teacher trio, 50:50 | 0.68316743 | +0.00002562 vs old student; CI [−0.00138485, +0.00139836] |

These records support that this three-seed teacher trio is useful for this
fixed F0 target and mixture; they do not establish that any weaker teacher
family is useful in every student or fold. See
[F0_EQUAL_TEACHER_TRIO_GRADE_R1.json](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json).

The preregistered fixed half screen combined the new clean-T3 CP5 student
with the old Z4 student and compared it only against that old student. Blend
WP was 0.6832549498 versus 0.6831418134, delta +0.0001131364, with
whole-component 95% interval [−0.00128701, +0.00151656] over 141 components
(800 bootstrap replicates). This is a small uncertain conditional result on
exposed F0. Both arrays are float32 storage, but new inference was FP32 and
old inference FP16; ancestry and recipe also differ. There is no recipe-winner,
public-transfer, or cross-fold claim. See
[the preregistration and result handoff](../../../../campaign_2026-10-12/clean_t3_f0_student_contribution_screen_r1/README_R1.md).

On F1, old/new student half scored 0.67182 versus new student 0.67103,
one pipeline seed and 457 exposed groups/149 components, with a broad interval
crossing zero. A completed F1 HFT comparison gave 0.67009 for the half blend
versus 0.67038 for the student; an extra-head blend scored 0.67188 versus
0.67182, a tiny uncertain difference. The direction changes across folds.
These are not independent lineage replication and not evidence to select from
F1 alone. See [F1 fixed half](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/old_new_f1_fixed_half_20261012_r1/RESULTS_R1.md)
and [F1 HFT comparison](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/oldz4_f1_hft_export_review_20261012_r1/RESULTS_R1.md).

The H382 cross-fold epoch work shows a fold-dependent four- versus three-epoch
effect: approximately +0.00078 in F0, about −0.0007 in F2, and F3 0.70865 at
epoch 3 versus 0.70657 at epoch 4; F1 comparisons also vary by fixed mixture.
Each fold/recipe has one seed except the separately reported F1 seed set, and
epoch changes also change total OneCycle schedule horizon. Results are from
exposed panels and do not imply a universal best epoch. Primary evidence is
[H382 cross-fold grade evidence](../../../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H382_CROSSFOLD_GRADE_EVIDENCE_R1.json).
The H384 F1 three-seed teacher mean was 0.67068046, equal trio 0.67254162,
and half H256 plus H384 trio 0.67264009; the incremental +0.00009847 is
exposed fixed-prediction evidence, not a new training seed or fresh fold.
Related source records are in
[IDEA_LEDGER_20261011.md](../../../../IDEA_LEDGER_20261011.md) and H384 receipts in the
October-11 operations continuation.

The large-teacher/native Q8 endpoint, clean-T3 pipeline, all-data H390 package
members, and historical old Z4/Z5 students are different source families.
Their scores should not be pooled without matched ancestry and precision.
Specific fit membership, epoch, parent checkpoint, teacher and initializer
provenance are in linked receipts rather than reconstructed from names.
