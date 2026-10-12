# Final handoff narrative review — R1

**Verdict: supported with two citation/status errata; archive remains pre-final.**
This is a static review of the narrative prefix (lines 1–492) in Main
`research/RESEARCH_COMPLETE_HANDOFF_20261012.md` at snapshot `42d3de59`.
I read linked Markdown/JSON receipts and the metric/trainer source only. I did
not read weights, NPY/cache contents, run benchmarks, or contact a provider.
The S4 local continuation was still draining at review time, so this review
does not mark the archive final.

The metric paragraph is accurate: the official `global_wp_clip2_v1` uses the
fixed `need_prediction AND is_scored` mask, float32 input conversion, ±2
clipping, and weights `abs(clipped label)`; it separately correlates each of
the two targets and averages them. This correctly distinguishes official
scoring from training-target clipping. See the
[metric implementation](../../src/wnn/gmetric.py#L20-L38) and the
[CP5 independent grade review](clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md).

The alpha interpretation is sound. Alpha 1.0 removes the true label from the
primary *target mixture*, but the training loss still weights rows by clipped
true-label magnitude and retains the auxiliary label-based objectives. That
is not a wholly label-free objective; the strict label-free experiment remains
untested. The source's target and weight operations confirm this distinction
in [`train_rnn.py`](../../scripts/train_rnn.py#L2575-L2603), while the
[exploitation plan](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/IDEA_STATUS_AND_EXPLOITATION_PLAN_20261011_R1.md#L68-L72)
states the experimental interpretation.

**Citation erratum:** the handoff's alpha paragraph points to LOOP_LOG passes
120/125 for the alpha-1 value. The exact `dm_a10_e6` 0.67068 entry is in
[Pass 128](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md#L589-L591).
Pass 125 records the alpha .9/.7 ladder and its uncertainty, not the alpha-1
measurement ([Pass 125](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/LOOP_LOG_20261010.md#L577-L579)).
Please update that citation at final regeneration. The narrative's values and
qualification are otherwise consistent with the linked deep dive.

The CP5 ancestry and target arithmetic match the linked review: seed 3, fixed
epoch 5, own `c20261003_combo_s3_f0@ep3` initializer; own H384 Muon teacher
seeds 3/4/5 at epoch 2; evaluation groups excluded from fitting; stored
target `float32(0.8125 * float32_mean(teacher predictions))` without cache
clipping; then `.1*clip(y)+.9*clip(Tstored)` in training. The resulting
nominal pre-clipping factor is .24375 per teacher, while clipping makes the
operation nonlinear. The generic `.1` parent-provenance rows are not the
numeric target coefficients. The old Z4 endpoint comparison is appropriately
qualified as confounded by target/parent/recipe/inference precision, not an
isolated causal result. The
[independent CP5 review](clean_t3_f0_cp5_grade_review_r25/INDEPENDENT_REVIEW_R1.md)
and [grade handoff](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/student_fit_r2/CONTROL_CP5_FULL_VALID_GRADE_HANDOFF_R1.json)
support these statements.

The Q8/Q4 and serving caveats are appropriately bounded. The package review
binds the exact A/B/C Q8 ZIP members and source hashes, while explicitly making
no score or held-out claim; it records local Intel callback tests, not an
EPYC guarantee. The portal scores and wall times are labeled as user-reported.
The separate full-fold Q8 result is not attributed to A/B/C. Q4's −0.00180125
result is correctly described as a small Q8-backend rounding screen, with no
native INT4 speed or full-validation evidence. D/E remain technically
reviewed but quality-unmeasured; E's 50.9-minute figure is explicitly a
capacity heuristic, not a measured portal runtime. See the
[independent A/B/C package review](../../artifacts/submission_review_10_12_best_r1/SUBMISSION_PACKAGE_REVIEW_R1.json),
[Q4 screen receipt](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/h390_submission_review_r4/FP32_Q8_INDEPENDENT_REVIEW_R1.json),
[D/E technical review](independent_reviews/submissions_10-12-best-r2/D_E_TECHNICAL_REVIEW_R1.json),
and [paired runtime receipt](z5_blend_runtime_pair_r1/PAIRED_RUNTIME_REVIEW_R1.json).

**Status erratum:** R7 is no longer one of two pending logical runs. Its
terminal receipt records provider `ERROR`, pre-fit literal `{python}` argv
failure, 114/114 collected files totaling 1,363,683 bytes, zero GPU use, and
no fit/export/prediction/score. This is infrastructure failure, not a model
result, and the stop forbids retry. The top paragraph gives these facts but
still says “two … runs pending”; replace that with “R7 is terminal and
collected; only local S4 remains pending.” The
[R7 terminal triage](../campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261012/r7_terminal_triage_r1/TERMINAL_TRIAGE_R1.json)
supports the correction. Keep the archive marked **pre-final** until root
records S4's terminal artifacts and refreshes the final snapshots.
