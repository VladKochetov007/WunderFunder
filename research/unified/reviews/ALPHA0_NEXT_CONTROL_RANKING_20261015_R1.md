---
kind: review
status: complete
created_utc: 2026-10-15
updated_utc: 2026-10-15
hypothesis_ids: [H-FROZEN-PARENT-ANCHOR-20261015, H-KD-CONTINUATION-VALUE-20261015, H-STUDENT-PARENT-GROWTH-20261015]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_FIT_EVAL_DIAGNOSTIC_R1.json
  - research/unified/ideas/FROZEN_PARENT_ANCHOR_CONTROL_20261015_R1.md
  - research/unified/reviews/ALPHA0_WARMPARENT_CP3_CP5_GRADE_REVIEW_20261015_R1.md
---

# Next control after the alpha-zero trajectory

## Finding

Prioritize the frozen-parent functional anchor against the shared T3 control after the active C/G/F/W profiles and cold-shape pair finish. It is the clearest next test of whether current T3 targets add useful information beyond constraining a student to its inherited function. Do not infer a teacher ceiling, student-capacity ceiling, or general KD effect from the current trajectory.

The source-bound diagnostic recomputed `global_wp_clip2_v1` for the same seed-3 warm H256×2 parent, alpha-zero continuation, and alpha-.9 T3 continuation. Alpha-zero minus alpha-.9 WP on the 1,405 fitting-valid groups is −0.000871 at CP1, +0.007670 at CP3, and +0.012247 at CP5. On the 468 excluded but repeatedly exposed F0 evaluation groups, the corresponding differences are −0.003947, −0.008744, and −0.014313. The CP5 conditional component interval is [−0.031167, −0.001122]; it describes this fixed exposed-panel prediction pair, not training-seed or public-transfer variation. The arms share an already-distilled parent and auxiliary objectives, so alpha zero does not remove all teacher ancestry.

The position bins are descriptive, not independent evaluations. At CP5 the fitting-valid target RMS is about 1.020/1.032 for T0/T1, versus 1.074/1.131 on evaluation. The difference is concentrated in early evaluation positions: rows 0–2,000 have target RMS 1.296/1.671 and means −0.080/−0.402, versus fitting-valid RMS 1.013/1.034 and means −0.182/−0.212. Later evaluation bins mostly have RMS near 1.0–1.13. Alpha-zero predictions also exceed |2| more often than alpha-.9, but these events remain rare: CP5 aggregate fractions are 0.0507%/0.0483% on fitting-valid and 0.0437%/0.0400% on evaluation for T0/T1, versus 0.00022%/0.00026% and 0.00012%/0.00086% for alpha-.9. The largest alpha-zero evaluation fractions occur in different bins by target (roughly 0.107% T0 and 0.110% T1 at rows 2,000–4,000); the high-target-RMS first bin is not where the largest prediction-excursion rates occur. Thus rare clipping excursions alone do not explain the WP gap. No residual or per-bin WP decomposition was run.

## Ranked next tests

1. **Frozen-parent functional anchor versus T3.** Add one arm to the shared two-epoch C control: use the exact same initializer, fitting membership, feature normalization, order, optimizer, batch/chunk geometry, auxiliary losses, target clipping and checkpoint rule, changing only the fixed target predictions from current T3 to the frozen parent. Reuse C as the comparator rather than refitting a second control. If parent-target S is competitive with T3 C, preservation of the inherited function may explain much of the continuation value; if T3 is clearly better, the current T3 target package adds value relative to this parent-target package. Neither result establishes a teacher-family ceiling. The parent and T3 targets differ in calibration as well as information: the existing T3 cache stores `0.8125 × mean(three teachers)`. The proposal itself notes this confound. Bind exact clipping/rounding and report fitting-only channel scale statistics; if a calibration-matched follow-up is needed, estimate it only on fitting members and freeze it before evaluation. Keep the parent's older distilled ancestry explicit. The cheapest useful run is this single S fit with the already-shared C result, after a bounded causal full-sequence parent-output/cache profile; the proposed fit-target cache is about 1.91 GB before metadata, so stream it to physical disk. The cost is an estimate, not a measured resource receipt.

2. **Stronger old-parent initialization under the same current T3 recipe.** This is a practical candidate path and can test whether a better inherited function raises the current endpoint without changing the target source. Reuse the same current T3 cache and match the active C geometry, membership, horizon, and evaluation. Existing old-Z4 parent score 0.682823 versus current parent 0.680941 is motivation only: their prior teacher/cache, normalization, Muon scope, update count, and precision differ. A result measures the value of those complete initializer packages under continuation, not a pure initialization effect or teacher-family effect. Do not alter the active C/G/F/W common parent.

3. **Label-only regularization or schedule rescue.** The existing alpha-zero curve already shows a large fitting-valid/evaluation divergence by CP3/CP5; shortening the schedule alone is not yet supported as a rescue because CP1 is below both the alpha-.9 child and its parent. A new regularizer or schedule could reduce overfitting, but changing several knobs together would not say whether external targets supplied information or merely stabilized optimization. If pursued after the anchor, preregister one change at a time against the same alpha-zero control and report fixed checkpoints. Preserve the possibility that label-only can improve under a different schedule; this trajectory does not reject that family.

## Scope and sources

The fit-valid metric is in-sample because those groups were included in fitting. The 468-group evaluation panel was excluded from this fit but is exposed development evidence. Position bins share sequences and are not extra folds. The diagnostic streams full prediction files and reproduces saved evaluation grades; it does not bootstrap bins or measure training variation. No new fit, payload load, or independent grading was performed for this review.

Sources: [machine-readable diagnostic](../../campaign_2026-10-15/student_quality_tradeoff_r1/ALPHA0_FIT_EVAL_DIAGNOSTIC_R1.json), [anchor proposal](../ideas/FROZEN_PARENT_ANCHOR_CONTROL_20261015_R1.md), [matched CP3/CP5 review](ALPHA0_WARMPARENT_CP3_CP5_GRADE_REVIEW_20261015_R1.md), and [quality bottleneck synthesis](../ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md). The active C/G/F/W and cold-shape recipes remain unchanged; root owns subsequent admission.
