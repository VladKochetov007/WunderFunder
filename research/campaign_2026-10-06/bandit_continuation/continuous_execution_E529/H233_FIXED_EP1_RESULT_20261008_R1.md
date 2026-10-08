# H233 fixed-epoch-1 exposed discovery result

Independent preregistered grade completed 2026-10-08T09:22:16.839824+00:00. Metric `global_wp_clip2_v1`,468 excluded groups/141 components; no fitting-group rows, altered warmup or scoring-mask relaxation. Fixed epoch1, all four heads and every declared blend weight reported. Parent numerical anchor parity was exactly zero; all artifact hashes, source-qualified row order and masks checked. This exposed local result supplies no public forecast.

| Readout | WP | Delta versus trained hidden-only |
|---|---:|---:|
| hidden_only | 0.681617903425 | +0 |
| additive | 0.681617399187 | -5.0423861897e-07 |
| multiplicative | 0.681617205539 | -6.97885718659e-07 |
| conditional_residual | 0.681618158484 | +2.55058394183e-07 |

Frozen C1 parent WP0.6816181421809264. Conditional-residual standalone delta versus parent is+1.6302588e-8; versus trained hidden-only+2.5505839e-7, conditional component-bootstrap95%interval[-4.24465e-8,5.70085e-7]. These tiny observed differences do not establish a useful submission gain; an interval crossing zero is not the allocation rule.

| Readout | Readout weight | WP | Delta versus frozen parent |
|---|---:|---:|---:|
| hidden_only | 0 | 0.681618142181 | +0 |
| hidden_only | 0.1 | 0.681618118382 | -2.37985208207e-08 |
| hidden_only | 0.25 | 0.681618082881 | -5.93003488536e-08 |
| hidden_only | 0.5 | 0.681618022999 | -1.19181470692e-07 |
| hidden_only | 1 | 0.681617903425 | -2.38755806459e-07 |
| additive | 0 | 0.681618142181 | +0 |
| additive | 0.1 | 0.681618067995 | -7.41859009779e-08 |
| additive | 0.25 | 0.681617956623 | -1.85558102661e-07 |
| additive | 0.5 | 0.681617770914 | -3.71266566934e-07 |
| additive | 1 | 0.681617399187 | -7.42994425429e-07 |
| multiplicative | 0 | 0.681618142181 | +0 |
| multiplicative | 0.1 | 0.681618165685 | +2.35045033437e-08 |
| multiplicative | 0.25 | 0.681618154792 | +1.26107390086e-08 |
| multiplicative | 0.5 | 0.681618008359 | -1.33822174231e-07 |
| multiplicative | 1 | 0.681617205539 | -9.36641525118e-07 |
| conditional_residual | 0 | 0.681618142181 | +0 |
| conditional_residual | 0.1 | 0.681618143807 | +1.62632196599e-09 |
| conditional_residual | 0.25 | 0.681618146326 | +4.14459377929e-09 |
| conditional_residual | 0.5 | 0.681618150194 | +8.01354116398e-09 |
| conditional_residual | 1 | 0.681618158484 | +1.63025877242e-08 |

No weight or checkpoint selected after seeing scores. Exact recipe is not promoted for packaging; family remains eligible. Prioritize faithful heterogeneous Muon teacher stages and capacity screen while a bounded independent audit checks whether near-anchor behavior follows scalar corrections, training-target treatment or an implementation issue. Reopen contextual readout on concrete new representation/objective/interaction, rather than repeatedly rerunning this recipe.

Delivery:29,840updates/0skips/1991.73s fit; export-only guard repair retained fitted checkpoint/source,131.20s exporter/136.55s controller, zero current-run OOM/kill. Parent was frozen FP16; heads/export arrays FP32. This is not a matched full-parent retraining objective and cannot attribute the effect solely to architecture.

Result SHA256`95d91cebfc19be05d607df2833523830553bfbe00810657c353ec3a6eb41b432`. [Original report](H233_FIXED_EP1_RESULT_20261008_R1.json); [root artifact admission](H233_FIXED_EP1_ROOT_GRADE_ADMISSION_20261008_R1.json); [independent grader review](reviews/H233_READOUTS_GRADER_SOURCE_REVIEW_20261008_R1.md).
