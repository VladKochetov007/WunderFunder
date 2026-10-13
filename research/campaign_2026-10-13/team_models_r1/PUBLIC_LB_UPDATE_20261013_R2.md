# Public leaderboard update — October13

The user supplied five scored Team kotiki rows, with quota 2/5 today and 90 total submissions. [Exact transcript receipt](PUBLIC_RESULTS_20261013_R2.json) records portal IDs, timestamps, scores, duration and memory. These are user-reported portal observations; they were not independently queried, and the portal payload hashes are unavailable. Original [64-row transcript](public_results.json) remains unchanged.

| Submission | Portal ID | Public score | Duration | Memory |
|---|---|---:|---|---:|
| kotiki_xteam_s6_z4_h390s3_q8_20261013.zip | 0mbzevhv | 0.6679 | 46m51s | 650MB |
| E_c75_plus_z5_q8_20261012.zip | 0j1f5ipt | 0.6675 | 51m22s | 651MB |
| xteam_h4.zip | 1hl88h25 | 0.6681 | 50m16s | 650MB |
| E_c75_plus_z5_q8_20261012.zip | 23wer4fu | 0.6675 | 49m16s | 651MB |
| h390_s3_s4_equal_q8_20261012.zip | 0rg9ujj5 | 0.6656 | 24m31s | 632MB |

The new seed6 cross-team blend scores **0.6679**, improving on E by 0.0004, trailing original seed5 xteam_h4 by 0.0002, and trailing the previously reported pair_ftt2 team best 0.6682 by 0.0003. It supplies further public evidence that mixing these team model families can improve on our E endpoint, without establishing superiority to the teammate ensemble or a general seed effect. This one public probe does not justify tuning subsequent coefficients from leaderboard feedback.

The pending E submission 0j1f5ipt is now scored 0.6675, equal to its earlier submission 23wer4fu. Its reported duration changed from 49m16s to 51m22s. The new blend completed in 46m51s, leaving 13m09s below the 60-minute limit, using 650MB. It was 3m25s shorter than original xteam_h4. Because runtime binaries are identical and these are different submissions with unknown host workload, these durations do not isolate an inference-engine speed gain.

The [checked seed6 release](../../../submissions/10-13-team-r1/README.md) remains in repository artifact storage. Source-qualified CV and serving comparisons remain the basis for future model selection; final-weight supplied-validation scores are diagnostic. E-TEAM-XTEAM-S6-RELEASE-20261013-R1 / H-TEAM-NATIVE-20261013.
