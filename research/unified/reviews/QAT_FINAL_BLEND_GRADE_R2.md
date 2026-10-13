# Independent final QAT blend grade R2

The three predeclared four-member blends were graded from the six terminal native prediction panels on the hash-pinned, complete F0 group order. On this exposed fitted-data diagnostic, both LSQ-W4 replacements trail the four-member reference; seed 3 has the smaller point-score loss. This does not establish a clean validation result or determine the root controller's public-test package choice.

| Fixed blend | Members, in native sum order | t0 | t1 | WP |
|---|---|---:|---:|---:|
| Baseline | common0, common1, common2, common3 | 0.6969958865293817 | 0.7075365888544030 | 0.7022662376918923 |
| QAT S3 | common0, common1, common2, lsq_s3 | 0.6965486179890060 | 0.7069797606392235 | 0.7017641893141148 |
| QAT S4 | common0, common1, common2, lsq_s4 | 0.6960108268278791 | 0.7064433872443293 | 0.7012271070361042 |

QAT S3 minus baseline is `−0.0004472685403756893` on t0, `−0.0005568282151794390` on t1, and `−0.0005020483777775642` on WP. QAT S4 minus baseline is `−0.0009850597015026440`, `−0.0010932016100736464`, and `−0.0010391306557880897`, respectively. S4 minus S3 is `−0.0005370822780105255` WP. These are point differences; this grade does not include a component bootstrap or confidence interval.

The recipe fixes each coefficient to float32 `0.25`, accumulates multiply-then-add in native member order, sanitizes a nonfinite blended total to zero, then applies the official `global_wp_clip2_v1` metric and clipping. The scorer reads only `seq_ix`, `step_in_seq`, `need_prediction`, `is_scored`, `t0`, and `t1` from the 468 selected validation row groups; it checks full 20,000-step sequences and the official prediction warm-up. The six native result receipts and prediction hashes, archive hashes, reset replay, shapes, data/split identity, member scores, and complete blend outputs are bound by the [machine receipt](receipts/QAT_FINAL_BLEND_GRADE_R2.json). The blend arrays are saved under `artifacts/qat_renewable_r1/final_blend_grade_r2/` for the separately owned ZIP callback-parity check.

R1 was frozen with the same three blends, but its first grader attempt stopped before admission/result validation or prediction access because the recipe omitted the literal `evaluation_groups` list expected by the consumer. R2 adds that list from `load_campaign_split` on the exact SHA-pinned `fold0.json`; it does not change members, coefficients, labels, or metric. During R1 diagnosis, the `common0` result metadata (including its standalone producer score) was inspected to locate the producer's `groups` field. No blend score was computed or read before freezing the R2 recipe. The R2 grader compares all six producer group arrays with the fixed split order.

The evaluation is the same exposed F0 panel used in model fitting/adaptation and earlier research (`468` sequences, `808,863` scored rows); it is neither clean cross-validation nor a public-score forecast. No uncertainty interval is reported. The validation parquet was not whole-file rehashed during grading: the producer data SHA receipt, fixed split hash, result identities, and selected row-group checks were used. Root retains final blend/package selection and must complete exact ZIP callback parity before any public test.

Frozen inputs: [original blend recipe R1](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/FINAL_BLEND_GRADE_RECIPE_R1.json), [schema-corrected recipe R2](../../campaign_2026-10-13/qat_renewable_r1/final_release_r1/FINAL_BLEND_GRADE_RECIPE_R2.json), [R2 grader](../../../scripts/grade_qat_final_blends_r2.py). `E-QAT-FINAL-ADAPTATION-20261013-R2`; R2 recipe SHA256 `01d93578e10de84eadaee659032552225216a5cc9201ac0aab55c25c59581315`, grader SHA256 `2d1f92cf66fccc7ed821bd7ce0a7a284776a2bf5ecbaa5227b943847c3fe14d4`.
