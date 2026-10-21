---
kind: review
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
---

# Sequence-level expert preference result review R1

## Question or claim

Does the completed early-to-late cached-expert diagnostic support persistent
sequence-specific expert preference or a separate B300 feature-router draft?

## Mechanism and method

The independent reviewer checked the frozen recipe, machine-readable grade,
resource receipts, and prior static review. It did not edit or rerun the
grader and did not inspect target or prediction payloads. It checked the late
score deltas and paired intervals, selector agreement against a
marginal-rate baseline, exposure and label-availability limits, and resource
outcomes.

## Evidence and interpretation

**PASS, with one interpretation correction.** The early-label selector's late
WP is 0.693871: −0.003798 versus TL3 (95% paired component interval
[−0.005690, −0.001453]) and −0.000587 versus fixed 75/25 (interval
[−0.002360, +0.001586]). These differences and intervals match the frozen
result.

Observed early/late expert-choice agreement is 68.16% per target. From the
reported marginal R3-choice rates, independent early and late choices would
agree 70.16% for target 0 and 68.64% for target 1. Raw agreement must not be
presented as evidence of persistence beyond the dominant TL3 choice; these
baseline values are descriptive and were not a predeclared significance test.

The profile and grade attempts both exited 0 under 2 GB reservations. Their
cgroup peaks were 533,319,680 bytes and 387,665,920 bytes; grade process RSS
was 569,810,944 bytes. Worker and parent-hierarchical OOM deltas were zero.
The different cgroup and process RSS scopes are retained as separate
measurements.

## Limits and validity

This is a fixed-cache diagnostic on repeatedly exposed F0. TL3 ancestry is
unknown, R3 was selected on F0, and the selector uses labels unavailable to
the test callback. Component intervals condition on these endpoints and
choices. No clean-fold, independent fitting-seed, mature-feedback, or public
transfer evidence follows. The reviewer considers the no-new-draft decision
appropriate: the frozen selector has no late advantage and no separate
label-blind signal predicts useful choices.

## Decision and next step

No separate B300 expert-router PR is justified by this screen. Keep the
existing retained-raw Kalman teacher/student PR #17 separate and unresolved;
this result neither validates nor invalidates that learned-feature path.

## Source records

- [Experiment packet](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/README.md)
- [Machine-readable result](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/result_r1.json)
- [Recipe](../../campaign_2026-10-18/online_features_r1/sequence_oracle_r1/recipe.json)
