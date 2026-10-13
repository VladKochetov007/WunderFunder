---
kind: review
status: inconclusive
created_utc: 2026-10-13T19:10:00Z
hypothesis_ids: [H-QAT-W4-20261013, H-QAT-LSQ-20261013, H-QAT-A8-20261013]
experiment_ids: [E-QAT-WEIGHT-100PCT-F0S3-20261013-R1, E-QAT-LSQ-10PCT-F0S3-20261013-R1, E-QAT-LSQ-10PCT-F0S4-20261013-R1, E-QAT-A8-BACKEND-20261013-R1]
---

# QAT progress and competition applicability — October 13

## Question and judgment

Today's results support QAT as a promising way to preserve quality when compressing recurrent weights. They do not yet establish a stronger complete submission or a public-score increase. This is the controller's synthesis of the linked independent grades, not a new independent review.

## Completed work

We completed matched floating continuation, uniform recurrent W4 QAT and mixed input-W4/recurrent-W8 QAT at 10%, 30% and full fitting coverage. All nine fits have complete native F0 evaluations. The seed-3 learned-scale comparison adds two completed fits and native grades; its fixed control reproduces the earlier W4 archive exactly and is not an independent replication. A separate seed-4 learned/fixed pair also completed all 1,440 updates per arm, with native evaluation in progress. These are 13 completed training runs, including the exact-control reproduction.

The versioned quantization format now preserves exact trained codes and scales through native serving. Packed-Q8 reproduces the original native-Q8 full prediction file byte for byte. Quantizer, checkpoint, export and complete-sequence reset/replay checks passed. Large-model activation-aware training remains an engineering investigation: the repaired small compiler probe passed, but the actual B16/C1000 four-update probe failed its validation gate despite reporting finite updates and changed represented weights. Its failure is under diagnosis; it is not a negative model-quality result.

## Measured improvements

PTQ means quantizing a floating continuation after training. Each primary comparison uses its matched fitting budget and the same serving format.

| Comparison | Control WP | QAT WP | Difference | Approximate 95% paired component interval |
|---|---:|---:|---:|---:|
| Full-data uniform W4 QAT vs W4 PTQ | 0.680741171 | 0.681362789 | +0.000621618 | [−0.000401165, +0.001686432] |
| Full-data mixed W4/W8 QAT vs mixed PTQ | 0.680660270 | 0.681420269 | +0.000759999 | [−0.000013193, +0.001655967] |
| 10% learned-scale W4 QAT vs matched nonlearned-scale QAT | 0.681298282 | 0.681492421 | +0.000194139 | [−0.00015310, +0.00053581] |

The full-data floating continuation served as Q8 scores **0.681605037**. Thus full-data W4 and mixed QAT remain below that reference by 0.000242248 and 0.000184767, respectively. Relative to their own PTQ compression losses, QAT recovers approximately **72% and 80%** in point estimates. Those ratios are descriptive calculations, not independently established effect sizes.

Mixed QAT has positive point deltas at all three fitting budgets: approximately +0.000139, +0.000271 and +0.000760. These share a parent, evaluation fold and seed; they are not three independent replications. Coverage and optimizer-update counts change together, so they do not isolate a scale-response effect. Uniform W4 was slightly negative at 10% and positive at 30% and full data.

The highest point score in this QAT comparison cohort is still the ordinary 10% floating continuation served as Q8, **0.681818498**. Its +0.0006831 versus the unchanged Q8 parent also has an interval crossing zero. QAT should not receive credit for gains from ordinary continuation.

## Significance and limitations

Each graded endpoint replays 468 full 20,000-row sequences on the same exposed F0 panel, with 808,863 officially scored rows. Uncertainty resamples **141 connected components**, not hundreds of thousands of supposedly independent rows. The intervals use 10,000 paired resamples and the saved raw-sum approximation; they are descriptive and conditional on fixed fitted models. Every primary improvement interval above includes zero. They neither establish reliable positive effects nor prove the mechanisms ineffective.

The graded QAT effects currently have one fitting seed. The seed-4 LSQ result is pending; F3 and broader fold confirmation are incomplete. Parent/teacher ancestry is audited against evaluated fitting groups, but prior evaluation exposure and adaptive research selection remain. Component resampling does not include training-seed variation or public-domain shift. No numerical public-score forecast or LB-position prediction follows.

## Practical applicability and next decisions

For the unchanged-parent model archives, packed Q8 is 1,086,824 bytes, uniform W4 is 678,883 bytes and mixed is 882,966 bytes: **37.5% and 18.8% smaller**, respectively. These percentages apply to that model archive, not an entire ensemble submission. All formats expand to the same resident byte-kernel weight storage. There is no demonstrated INT4 execution, resident-memory reduction or CPU speedup; local concurrent callback times are not controlled runtime comparisons.

Compression could help the leaderboard if it allows a demonstrably stronger fixed ensemble or larger student to fit the package budget while meeting runtime limits. It does not yet provide the runtime headroom for an extra member. A direct replacement of the matched Q8 model is not supported by today's point scores. Mixed precision is the leading quality-preservation candidate; uniform W4 saves more archive space. Neither dominates the complete quality/size/runtime decision.

Next: finish seed-4 grading; diagnose and validate the activation-aware profile before fitting; repeat promising recipes on source-qualified folds and seeds; then compare compressed replacements and complementary additions inside complete fixed blends. Keep QAT as one useful research direction alongside improvements to the underlying predictor. Existing evidence does not establish progress to 0.700 or any particular public rank.

The corrected private Kaggle seed-4 profile was pushed once as R2 version 1 and was QUEUED at dispatch. Its R1 predecessor failed resource admission before candidate fitting. The actual-data TPU bridge remains queued at the latest observation. Neither cloud job contributes a completed quality result to this review.

## Source records

- [Authorized plan and campaign](../../campaign_2026-10-13/qat_renewable_r1/README.md), including the append-only experiment/allocation records and frozen sources.
- [Unchanged-parent baseline and archive sizes](../../campaign_2026-10-13/qat_renewable_r1/baseline_results_r2/README.md).
- [Independent full-data W4 grade](../reviews/QAT_100PCT_UNIFORM_W4_MATCHED_GRADE_R1.md) and [mixed W4/W8 grade](../reviews/QAT_100PCT_MIXED_W4W8_MATCHED_GRADE_R1.md).
- [Independent learned-scale grade](../reviews/QAT_LSQ_10PCT_PAIRED_GRADE_R1.md), [ordinary-continuation grade](../reviews/QAT_FLOAT_CONTINUATION_COMPRESSION_GRADE_R1.md), and [10%/30% wave](../../campaign_2026-10-13/qat_renewable_r1/MATCHED_ADAPTATION_WAVE_R1.md).
- [Seed-4 recipe and execution](../../campaign_2026-10-13/qat_renewable_r1/lsq_f0_s4_r1/README.md); native grading is pending at this snapshot.
- [Activation actual-data admission](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_ACTUAL_ADMISSION_R2.json) and [failed validation receipt](../../campaign_2026-10-13/qat_renewable_r1/ACTIVATION_ACTUAL_GPU_R2.json); engineering evidence only.
- [Kaggle R2 dispatch](../../campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/ROOT_KERNEL_PUSH_R2.json), [preserved R1 failure](../../campaign_2026-10-13/qat_renewable_r1/kaggle_f0_s4_r2/TERMINAL_PROFILE_FAILURE_R1.json), and [TPU bridge](../../campaign_2026-10-13/qat_renewable_r1/tpu_bridge_v8/README.md).
