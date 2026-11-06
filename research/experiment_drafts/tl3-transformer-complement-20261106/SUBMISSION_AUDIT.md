# Submission and experiment audit — 6 November 2026

The official [leaderboard](https://wundernn.io/connectome/leaderboard) and signed-in [team submission history](https://wundernn.io/connectome/submit) were read in Chrome on 6 November 2026. The history contained 108 entries: 102 numeric scores, five `SCORE:TIMEOUT` entries and one other unscored entry. These are displayed platform results, not independent replays. No ZIP was uploaded or model executed during this audit.

## Public frontier

| Rank | Team | Public WP | Submissions |
|---:|---|---:|---:|
| 1 | Gargantua | 0.67596 | 247 |
| 2 | baselinemax | 0.67586 | 11 |
| 3 | Order_of_the_Orb | 0.67567 | 106 |
| 4 | blanc | 0.67498 | 86 |
| 5 | Mech_Korolya_Artura | 0.67487 | 149 |
| 6 | air_chair | 0.67418 | 31 |
| 7 | opiumfive | 0.67380 | 209 |
| 8 | musya | 0.67323 | 75 |
| 9 | MRRRR | 0.67307 | 89 |
| 10 | cool_team_name | 0.67209 | 275 |
| 39 | kotiki (our team) | 0.66832 | 108 |

The displayed leader-to-team gap is 0.00764 WP. Competitors' code/weights are not exposed by this table; neither model family nor compute budget can be inferred from their rank or submission count. This review covers all available team history entries and the public top-ten table, not inaccessible competitors' ZIPs.

## Best team entries and latest submitted candidate

| ZIP | Displayed ID | Public WP | Duration | Date shown |
|---|---|---:|---|---|
| `tl3.zip` | `1lmehfi3` | **0.66832** | 39m12s | 15 October |
| `tl4.zip` | `219y3mlk` | 0.66830 | 50m7s | 15 October |
| `pair_ftt2.zip` | `03sqa1fi` | 0.66821 | 44m42s | 9 October |
| `xteam_h4_D_q8vnni_fused.zip` | `182a1r2g` | 0.66808 | 47m27s | 15 October |
| `xteam_h4_C_q8vnni_blocked.zip` | `2nk0qaui` | 0.66808 | 57m27s | 15 October |
| `xteam_h4.zip` | `1hl88h25` | 0.66808 | 50m16s | 12 October |
| `tl3_h390_s3_s4_20261024.zip` | `1ci60k81` | 0.66806 | 52m11s | 24 October |
| `pair_ftt1.zip` | `0eovqvkd` | 0.66806 | 34m57s | 9 October |
| `pair_lowlr.zip` | `00olgj1c` | 0.66800 | 45m7s | 14 October |
| `kotiki_qat_lsq_20261014.zip` | `00fpe72k` | 0.66797 | 54m26s | 13 October |
| `tl3_h390_s3_20261024.zip` | `1h34eouq` | 0.66792 | 41m11s | 24 October |
| `kotiki_xteam_s6_z4_h390s3_q8_20261013.zip` | `0mbzevhv` | 0.66788 | 46m51s | 13 October |
| `tl3_two_parents_20261022.zip` | `0r4h74xw` | 0.66771 | 28m31s | 22 October |
| `b300_h512_cp5_tl3_20261021.zip` | `0equo922` | 0.66667 | 53m47s | 21 October |
| `tabdata-w18-final-20261103-r1.zip` (latest) | `1h7kp3oq` | **0.66453** | 14m32s | 4 November |

The final W18 is a single all-labels GRU256×2, not the full new TL3+LSTM mixture. [PR54's published artifact report](https://github.com/VladKochetov007/WunderFunder/blob/6493556e853e02538d50baed049073eeb717e9d7/research/experiment_drafts/final-submit-all-labeled-20261031/RESULTS_20261103_W18_DELIVERY_R1.md) pins the 5,454,663-byte ZIP with SHA256 `932700e4e10721829641870f18198fca06ba3f9c5f06be5541d70bf38bf880f9`. Its GitHub upload receipt used uppercase ID text; this table preserves the browser's displayed lowercase text without guessing ambiguous O/0 transcription. The portal result is lower than the old TL3 by 0.00379. This is a negative transfer observation for that submitted recipe, not proof that all-data fitting or every fresh model is worse.

Several larger mixtures are near the one-hour inference limit, and five entries timed out. Training hardware does not increase the competition CPU serving allowance.

## Source-clean development results are a separate scale

| Fixed predictor / experiment | Official-valid WP | Evidence and limitation |
|---|---:|---|
| Fresh PR49 canonical TL3 | 0.6923774245652725 | [Exact recovery receipt](https://github.com/VladKochetov007/WunderFunder/pull/49#issuecomment-6532374408): 10,325 fitting train sequences, zero fitting valid, all 1,873 valid evaluated. Development-exposed. |
| 90% fresh TL3 + 10% LSTM | 0.692502063 | [PR53 final comparison](https://github.com/VladKochetov007/WunderFunder/blob/f98543d2d11b162a2e1004a968043cf9269f023a/research/experiment_drafts/multistage-optuna-b300-20261031/quality-execution-r1/RESULTS.md): +0.000124638; exposed and selected from several reported mixtures. |
| 90% fresh TL3 + 10% GALAM | 0.692496447 | [PR52 table](https://github.com/VladKochetov007/WunderFunder/pull/52#issuecomment-6532373933): +0.000119022; exposed, not a replicated public gain. |
| Fresh clean W18 singleton | 0.6914165059769162 | Same PR49 recovery, Torch output. This is not the final all-labels W18's independent score. |
| PR53 equal-four new models | 0.684699354 | [Complete primary results](https://github.com/VladKochetov007/WunderFunder/pull/53#issuecomment-6539567337): below TL3; 39 final artifacts do not mean 39 independent improved submissions. |
| PR61 Transformers | 0.6717207056 / 0.6729531606 | [Terminal report](https://github.com/VladKochetov007/WunderFunder/pull/61#issuecomment-6639856413): four-epoch recipe; two student transfers remain unstarted after a profile-identity mismatch. |

PR52's stronger standalone distilled students did not improve their fixed TL3 blends. PR53's additional GRU batch/history candidates also failed to improve those mixtures. This motivates training the Transformer for the **fixed final composition** rather than selecting a standalone score and assuming blend benefit. It does not prove that the new objective works.

## Code reuse and non-duplication

- [PR35](https://github.com/VladKochetov007/WunderFunder/pull/35) already implements typed numerical/event features; do not call event211 a new direction.
- [PR38](https://github.com/VladKochetov007/WunderFunder/pull/38) proposes a small frozen-parent residual head; this packet extends the additive-correction family to a large causal Transformer and directly matched mixture objective.
- [PR48](https://github.com/VladKochetov007/WunderFunder/pull/48) already covers Transformer width/context, [PR58](https://github.com/VladKochetov007/WunderFunder/pull/58) nested OOF mixtures, and [PR65](https://github.com/VladKochetov007/WunderFunder/pull/65)/[PR66](https://github.com/VladKochetov007/WunderFunder/pull/66) attention gates/SwiGLU. These remain separate unmeasured proposals.
- The open [mnemonik-dev/wunder](https://github.com/mnemonik-dev/wunder) workspace is for the current competition and describes a causal linear/GRU mixture. Its reported validation evidence is not a verified top-public solution. No reproducible implementation of the live top three was found in the bounded public GitHub/HF search; inaccessible solutions remain unknown.

The [gradient-boosting antecedent](https://doi.org/10.1214/aos/1013203451) motivates fitting an additive correction conditional on an existing predictor. The present Transformer architecture, fixed blend, curvature scaling and compute allocation are our experimental choices, not outcomes of that paper.
