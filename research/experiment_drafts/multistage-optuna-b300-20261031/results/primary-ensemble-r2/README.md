# Frozen ensemble: full official primary valid

| Candidate | WP |
|---|---:|
| direct_gru | 0.681170645 |
| direct_lstm | 0.680193898 |
| transformer | 0.675296541 |
| finetune_gru | 0.681492864 |
| finetune_lstm | 0.680248448 |
| kd_label_control | 0.675889945 |
| kd_student | 0.678291391 |
| direct_other | 0.680869802 |
| Fixed Stage4 ensemble | 0.683227279 |
| Fresh TL3 reference | 0.692377425 |

All1873 official-valid sequences and3,528,273 is_scored rows after99 warmup. New candidates fit8260 earlier official-train sequences; fixed TL3 reference0.6923774245652725 fits10325 and is not rescored in this package. This is an unmatched fitting-budget deployment comparison.

Frozen Stage4 weights were learned only on the exposed2065-member train-only control, with24 actual completed CPU Optuna trials. Completion of trial24 reused unchanged original scientific functions and preserved all first23 trial records; restarted sampler RNG is recorded in the recovery receipt. No primary-valid label selected weights.

The fixed ensemble scores 0.6832272791326639 vs TL3 0.6923774245652725; delta -0.009150145432608614. It does not improve the baseline. One inner-selected KD pair improves primary WP from0.675889945471552 to0.6782913914220583; that is one selected seed, not six primary replications.

Eight member inference outputs are retained. The initial parent failed only at CPU postprocessing because its file reader rejected28MB arrays; corrected cached-only blending reused all8predictions. Failed original receipts and successful cached-blend receipt remain separate. No neural training or GPU member scoring was repeated by that repair.

Four fixed direct-label configurations now have full-primary fresh-refit admissions. Their10325-member outcomes are separate future evidence; no new public score or submission is claimed.
