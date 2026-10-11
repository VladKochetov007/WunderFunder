Two strongest operational candidates are ready for human upload. All-data final weights have no held-out score; the ranking is provisional pending complete clean-fold pipeline comparisons.

1. `12_h388_teacher_trio_all_fp16store_q8vnni.zip`: equal own H384 seeds3/4/5; 13,735,320 bytes. Same-recipe full-fold float discovery .6725416214, 457 exposed groups /149 components. Selective float16 storage keeps Q8 runtime and loses .0000066413 against the same Q8 trio in the fixed41-sequence native panel.
2. `11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip`: half own H256 student/half H384 teacher;12,914,019 bytes. Related float prototype .6725742672, with parent/since-weight differences explicitly retained.

Actual Python3.11 Q8 selection, finite predictions and deterministic sequence reset passed independent gates. The native small panel slightly favors the teacher pair over the trio, whereas full-fold float evidence favors the trio; no universal ranking claim. The four-member prototype improves the trio on this panel, but does not certify the changed H390 final student. No numerical public-score forecast or .670+ expectation is established. Original ZIPs/metadata remain frozen.
