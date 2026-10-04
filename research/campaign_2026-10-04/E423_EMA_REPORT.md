# E423 EMA decay retained over reference rows

Fold0 seed3, four-epoch combo recipe, batch32/chunk1000/fixed epoch3. Intervention used `--ema-reference-rows 2000`, giving effective update decay sqrt(0.999)=0.9994998749; the newly trained same-source control used ordinary0.999. Same objective, source data, teachers and optimizer. Both epoch3 models were exported in common FP32 inference; paired components, n_boot2000,468 sequences/141 components.

EMA-in-rows WP0.6803669274; control WP0.6809410229. Difference−0.0005740955, 95% interval[−0.0017969793,+0.0003028590], p(delta<=0)=0.8645. Class0 declined−0.00218, class1−0.00019, class2−0.00034, class3 increased+0.00022. This one exposed fold/seed does not show a gain from retaining EMA decay in rows. The interval permits a small positive effect; it is not evidence of universal harm.

Training took2309s for EMA and2410s for the fresh control; common-FP32 prediction65.2s and62.1s; paired score70.8s. GPU peak was within4.8GB enforced stop; no OOM. These two arms consumed about1.31 device-GPU hours combined. No public submission or score.

Evidence: checks/e423_report.json, checks/ema_sequence_v1.json, sealed control/intervention/scorer specs and runner attempts. Original combo is descriptive; the inference uses the fresh same-source control.
