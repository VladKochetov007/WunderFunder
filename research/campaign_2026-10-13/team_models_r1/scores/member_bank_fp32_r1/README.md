# Complete FP32 member and fixed-mixture diagnostics

All 14 member exports and 9 predeclared mixtures were graded on 1,873 complete supplied-valid sequences,3,528,273 selected rows. Full prediction-file hashes, per-group hashes, terminal producer metadata, source dependencies and input order passed the independent grader. Official metric: global_wp_clip2_v1, blending before clipping.

**These final weights have fitted or unknown overlap with supplied validation. This is diagnostic evidence, not clean CV, and these FP32 outputs differ from served Q8 predictions.**

| Fixed endpoint | Diagnostic WP |
|---|---:|
| pair_ftt2_fp32 | 0.712025385 |
| E_fp32 | 0.712082246 |
| xteam_h4_fp32 | 0.712660191 |
| e6a_fp32 | 0.703243606 |
| distilled_pair_fp32 | 0.711826542 |
| ftt_pair_fp32 | 0.711079377 |
| pair_0.25_E_0.75_fp32 | 0.712359040 |
| pair_0.50_E_0.50_fp32 | 0.712426923 |
| pair_0.75_E_0.25_fp32 | 0.712308726 |

Original xteam_h4 FP32 exceeds E by 0.000577946 and pair_ftt2 by 0.000634806. This comparison is conditional on these final weights and exposed/fitted data. Our H390 seed3 alone is 0.713414409 on this diagnostic, so the cross-team mixture does not dominate all individual members. No public forecast follows.

The new exploratory seed6 submission was fixed before this grade but is not one of the nine original mixtures; this table does not assign it a score. No coefficients were optimized. The complete individual-member grades and clipped prediction/residual correlation matrices are in [RESULT.json](RESULT.json).

[Admission and source pins](../../MEMBER_GRADE_ADMISSION_R1.json), [execution log](../../MEMBER_GRADE_R1.log), [member bank terminal receipt](../../MEMBER_BANK_TERMINAL_R1.json), and [fixed original plan](../../MEMBER_BANK_PLAN_R1.json) preserve the comparison. E-TEAM-MEMBER-BANK-20261013-R1 / H-TEAM-NATIVE-20261013.
