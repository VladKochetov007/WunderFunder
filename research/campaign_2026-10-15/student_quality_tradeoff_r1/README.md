# Student quality tradeoff R1

This folder holds a source-bound, not-yet-admitted comparison of continued label supervision against continued clean-T3 distillation from the same warm H256×2 student parent. The proposed intervention changes only `--distill-alpha`; it does not test cold-start distillation or remove the initializer's earlier distillation ancestry.

- [Matched warm-parent alpha-0 packet](MATCHED_WARMPARENT_ALPHA0_PACKET_R1.md) specifies the exact checkpoint/cache/split/source pins, the single argument difference, two-epoch recoverable stages, fixed checkpoint evaluations at epochs 2/4/6, and the known resource bound. It authorizes no fit and requires fresh review/admission.
- [Machine-readable packet](MATCHED_WARMPARENT_ALPHA0_PACKET_R1.json) gives the frozen source/weight/membership/cache hashes, current alpha-.9 reference checkpoint hashes, stage boundary mapping, and resource estimates for later launch construction.
- [Quality bottleneck evidence synthesis](../../unified/ideas/QUALITY_BOTTLENECK_AND_STUDENT_TRADEOFFS_20261015_R1.md) is the campaign's current interpretation of the exposed-F0 student and teacher evidence; the [independent review](../../unified/reviews/CLEAN_T3_TEACHER_STUDENT_BOTTLENECK_REVIEW_20261015_R1.md) records the critic's source and comparison checks.

Status: prepared from metadata, source and existing receipts only. No new fit, prediction, grade, provider action or resource admission was performed.
