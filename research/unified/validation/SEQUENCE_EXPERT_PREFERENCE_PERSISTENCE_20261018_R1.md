---
kind: result
status: complete
created_utc: 2026-10-18T05:36:50Z
updated_utc: 2026-10-18T05:36:50Z
hypothesis_ids: [H-SEQUENCE-EXPERT-PREFERENCE-PERSISTENCE-20261018-R1]
experiment_ids: [E-SEQUENCE-ORACLE-PERSISTENCE-20261018-R1]
evidence_paths:
  - research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/recipe.json
  - research/campaign_2026-10-18/online_features_r1/sequence_oracle_r1/result_r1.json
  - research/campaign_2026-10-18/online_features_r1/runtime/attempts/sequence-oracle-profile-20261018-r1.json
  - research/campaign_2026-10-18/online_features_r1/runtime/attempts/sequence-oracle-grade-20261018-r1.json
  - research/unified/reviews/SEQUENCE_ORACLE_STATIC_REVIEW_20261018_R1.md
  - research/unified/reviews/SEQUENCE_ORACLE_RESULT_REVIEW_20261018_R1.md
---

# Early-to-late sequence-level expert preference R1

## Question or claim

Does choosing between frozen TL3 and R3 predictions for each sequence and
target using early labels preserve an advantage on later rows, enough to
motivate a feature-derived expert router or preference teacher?

## Mechanism and method

The frozen diagnostic uses all 468 F0 groups (141 connected components). It
selects TL3 or R3 by local official weighted-Pearson score on rows 99–4,999,
then holds that choice fixed on rows 5,000–19,999. Controls are TL3, R3, and a
fixed 75/25 TL3/R3 mixture; a late-label hindsight selector is descriptive.
The metric and mask are `global_wp_clip2_v1`. No model was fit.

## Evidence and interpretation

The early selector reached 0.714842 WP on its selection segment, only
0.000212 above TL3 there. Its later score was 0.693871:

| Late endpoint | WP |
|---|---:|
| TL3 | 0.697669 |
| R3 | 0.679802 |
| Fixed 75/25 TL3/R3 | 0.694457 |
| Early-label selector | 0.693871 |
| Late-label hindsight selector | 0.696703 |

The early-label selector was −0.003798 versus TL3, with a paired conditional
component-bootstrap 95% interval [−0.005690, −0.001453]. It was −0.000587
versus the fixed 75/25 blend, with interval [−0.002360, +0.001586]. The
hindsight selector also scored below TL3; it is not a formal upper bound under
the globally pooled metric. The first comparison is negative for this fixed
selector on this panel; the second remains uncertain rather than a proven
loss.

The selected R3 rate fell from 21.6%/24.8% early to 14.5%/13.0% late for the
two targets. Early/late identity agreement was 68.16% for both. As a
post-hoc descriptive baseline, independent choices with those marginal rates
would agree about 70.16% and 68.64%; raw agreement therefore does not show
positive persistence beyond the TL3-dominated base rates. This comparison has
no uncertainty interval and is not a significance test.

An independent result review confirmed the score and interval alignment and
requested that the marginal-agreement baseline accompany the result. The
pre-grade static review passed source, timing, metric, and bootstrap checks.

## Limits and validity

F0 is repeatedly exposed, R3 was selected using F0, and TL3 ancestry is
unknown. Choices use validation labels unavailable to the test callback, so
this is not an online policy. The component intervals condition on cached
predictions and selected early choices; they do not measure fitting seeds,
clean folds, feedback maturity, or public transfer. The late hindsight result
is not a formal pooled-metric oracle bound. The result narrows these two fixed
experts and this label-informed selector only; it does not reject broader
causal online learning or learned state features.

The eight-group memory profile processed eight F0 groups but materialized the
full target and mask arrays from the compressed validation NPZ. Both this
profile and the full grade ran under a 2 GB worker reservation with zero OOM
counter deltas. Profile cgroup peak was 533,319,680 bytes; grade cgroup peak
was 387,665,920 bytes and process RSS was 569,810,944 bytes. The profile's
whole-archive SHA matched again before the full grade.

## Decision and next step

Do not create a separate B300 expert-router draft from this result: the
label-informed selector has no later advantage, its agreement is explained by
the dominant TL3 choice, and no label-blind input signal has been shown to
predict useful choices. Keep the larger retained-raw Kalman feature
teacher/student proposal in existing [draft PR #17](https://github.com/VladKochetov007/WunderFunder/pull/17)
as a distinct, still-unprofiled direction. This screen does not close online
learning with verified mature feedback or causal learned-state features.

## Source records

- [Experiment packet](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md)
- [Frozen recipe](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/recipe.json)
- [Machine-readable result](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/result_r1.json)
- [Profile attempt receipt](../../campaign_2026-10-18/online_features_r1/runtime/attempts/sequence-oracle-profile-20261018-r1.json)
- [Grade attempt receipt](../../campaign_2026-10-18/online_features_r1/runtime/attempts/sequence-oracle-grade-20261018-r1.json)
- [Pre-grade static review](../reviews/SEQUENCE_ORACLE_STATIC_REVIEW_20261018_R1.md)
- [Independent result review](../reviews/SEQUENCE_ORACLE_RESULT_REVIEW_20261018_R1.md)
