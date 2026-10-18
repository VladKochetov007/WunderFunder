# Model families and training

Architecture, training lineage, distillation and retained checkpoint sources.

## Current navigation

- [Frozen zero-i1 replay](../../campaign_2026-10-17/research_reset_r1/bugfix/REPLAY_zero_i1_v2.md) — unchanged dftt_zi1_s3 checkpoint,4prespecified groups/11,522scoredrows, canonical WP0.06289742withwrong inputs versus0.72994782after restoring training preprocessing. This isolates an export defect; it is not a new model or full-panel ranking.

- [Current model and submission catalog](CURRENT_MODEL_CATALOG.md) — production packages, ancestry, exact ZIP identities and the distinction between research endpoints and serving engines.
- [Private model archive](huggingface/README.md) — run lookup and source-qualified checkpoint preservation.

For active status, start with [current state](../../../specs/current-state.md).
The dated entries below retain their original descriptions and links. Their
“pending”, “active” and authorization statements reflect their recording time;
use a later result receipt or the current audit to establish final status.

<details>
<summary>Dated index preserved before the October 17 maintenance</summary>

Use this domain for architecture, optimization, distillation, target construction, and model ancestry.

## Read these sources

- [Current T3 student versus exact parent, paired regrade](../../campaign_2026-10-15/student_quality_tradeoff_r1/CURRENT_T3_PARENT_PAIRED_GRADE_R1.json), E-KD-CURRENT-PARENT-PAIRED-GRADE-20261015-R1/H-KD-CONTINUATION-VALUE-20261015, reproduces official F0 scores .6811553603 versus .6809412068 and measures delta+.000214154 with conditional141-component interval[-.00338452,.00309511]. Existing prediction/checkpoint/sidecar and membership bindings passed the unchanged scorer. One exposed fold, unknown parent inference precision and changed continuation recipe prevent causal KD attribution; no training or public forecast. [Interpretation and next matched comparison](../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md).

- [Private Hugging Face archive and model lookup](huggingface/README.md) — organization model repository, checkpoint inventory, per-run descriptions and verified transfer receipts. Starts with public-best source students and strong local teachers/students, then preserves historical models with their repository paths and research IDs.
- [Model families, training, and source-qualified findings](../archive/2026-10-12/pages/02-models-and-training.md)
- [Current best solution explained](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/solution_report/CURRENT_BEST_SOLUTION_EXPLAINED.md)
- [Research methods](../../RESEARCH_METHODS.md) and the [model/experiment records](../../experiments.jsonl)

Every comparison should name independent weights, teacher and initializer ancestry, fitting groups, target treatment, optimizer, schedule, and actual completed epoch. Local score deltas do not directly forecast public score.

## Resumed strategy execution — 2026-10-13

[Strategy execution campaign](../../campaign_2026-10-13/strategy_execution_r1/README.md) implements the full October12 review under the user's subsequent explicit resume. It links the approved plan, full proposal coverage, resource authority, frozen comparisons and live frontier. All Kaggle allowance and actual local GPU capacity are authorized; CPU/RAM protections and no-paid-spend remain. Initial status is preparation, with no new fitted result. Existing folds remain exposed.

- [Clean F0 teacher trio and old student grade](../../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_local_equal_trio_grade_r1/F0_EQUAL_TEACHER_TRIO_GRADE_R1.json) is the October12 provenance-bound comparison underlying the October14 homogeneous teacher cache. Three scratch H384 seeds3/4/5 score .678989/.678367/.679698; their equal trio scores .680902. Historical oldz4_h230_student_f0_s3_r3 scores .683142 and its predeclared half/trio mixture .683167. One exposed F0 panel, teacher FP32 versus old-student FP16, shared ancestors where recorded; this is not broader-fold confirmation or a new ranking. The stronger old-student result remains a historical control, not omitted on public-score grounds; the current fresh-weight fixed-budget ladder measures architecture under common clean targets, not a matched claim against that older fit.

- [H512×1 matched student first screen](../../campaign_2026-10-14/cpu_student_frontier_r1/README.md#fourth-matched-screen-h5121) records fullF0seed3WP.673237761 and conditionalpaired+.00346012 overH256×2 withclean teacher/source ancestry and fixed14920updatebudget. Independentreview supports advancing the shape, not overallwinnerpromotion: native serving quality/cost and distinctseed/fold checks remain. Historical more-trained H256control is not displaced by this screen.

- [H384×2 first-screen grade and completion](../../campaign_2026-10-14/cpu_student_frontier_r1/results/student_frontier_f0s3_h384_l2_r2_fold0.json), H-STUDENT-LADDER-20261014, scoresWP.6716880442 versus matchedH256×2.6697776419, delta+.0019104024 with conditionalcomponent interval[.0003020132,.0034776104]. Complete ten-step sequence and grade validation passed; independent result review/native quantized grade pending. This is one exposedF0/seed3 shortfit and trails H512×1.6732377615 under the same recipe; both are development candidates, not yet new final submissions.

- [Clean F3 teacher and fixed older-model mixture](../../campaign_2026-10-15/f3_teacher_export_r2/README.md#root-official-f3-grading-after-collection), H-KD-COMPLEMENTARY-TEACHER-20261015/E-F3-TEACHER-FIXED-HALF-20261015-R1, supplies505-group/145-component exposedF3 evidence: teacher.708015 versus older.709542, fixedhalf.710417 (delta+.000875, component interval[-.000455,.002013]). All exports collected/finite/identity checked and scored with frozen helpers; one seed/differentrecipes/olderprecisionunknown, half chosen after singles. Supports retaining possible complementary teacher signal, not a new student/publicbest or cross-fold numerical comparison.

- [Derived F0 GRU growth initializer](../../campaign_2026-10-15/gru_parent_growth_r1/README.md) — parent c20261003_combo_s3_f0@ep3 expanded H256×2→H384×2 with pinned artifact/provenance and complete-sequence FP32 parity. Source-parent epoch is retained; no new updates, independent replication, native parity or measured quality. H-STUDENT-PARENT-GROWTH-20261015.

</details>

- [Causal smoother GRU teachers and B300 student handoff](../../experiment_drafts/kalman_teacher_gru_distill_b300/README.md) — code-only adapter and proposed fresh-teacher/student-cache comparison with raw-zero-pad, matched EMA, fixed local trend and learned local trend controls at pilot4/all112 scopes. Prediction reconstruction and checkpoint/cache provenance support are implemented; the B300 profile, clean four-family registry, student cache, fits and scores remain pending.
