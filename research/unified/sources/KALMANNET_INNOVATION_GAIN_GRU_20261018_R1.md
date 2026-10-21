---
kind: source
status: complete
created_utc: 2026-10-18T08:18:33Z
updated_utc: 2026-10-18T08:30:52Z
hypothesis_ids:
  - H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1
experiment_ids:
  - E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1
evidence_paths:
  - ../ideas/KALMAN_INNOVATION_GAIN_STATE_GRU_20261018_R1.md
  - ../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md
---

# KalmanNet-inspired innovation-conditioned GRU: source assessment

## Primary sources

- [KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043) motivates estimating a gain from state-estimation error and measurement innovation while preserving a predictor/corrector structure. Its reported tasks have an explicit state-space interpretation.
- [Recursive KalmanNet](https://arxiv.org/abs/2506.11639) explores recursive neural state-estimation updates. Its preprint results are not on this competition or its anonymous market-state regression targets.
- [KARNet](https://arxiv.org/abs/2305.14644) is a neighboring learned-filtering proposal and supplies no local transfer evidence.

## Transfer assessment

These papers motivate an operator-level hypothesis: predict a latent observation, measure its innovation, and learn a bounded correction gain. They do not show that this task's latent channels are physical states, that covariance is identifiable, or that the operator improves the competition's weighted Pearson objective.

The proposed adapter has no calibrated process/measurement covariance recursion, observed state labels, or inference-time weight update. It is accurately described as a **KalmanNet-inspired latent observer**, not a calibrated Kalman filter or test-time-learning system. A context-only gain control masks innovation from the gain network but still applies the gain-times-innovation correction; the contrast isolates the gain's innovation conditioning only.

Nearby local results are limited: fixed output filters lost on exposed F0; an additive groupwise-state ridge proxy lost on all-train rows; recurrent append deltas changed sign across two fit seeds on exposed F0; and a selected fixed local-trend input replacement showed a small uncertain positive point. These results motivate a distinct learned latent update but do not establish expected benefit.

## Evidence needed before interpreting a transfer claim

The local code review required prefix/chunk/reset/batch-isolation invariants, a full-shape strict-FP32 B300 profile, and a frozen clean per-cell teacher-source registry before any fit. A matched student comparison must keep the ordinary GRU endpoint, cache ancestry, coefficients and budget fixed. None has been run. No external data, pretrained weights, score, runtime or performance result is imported from these sources.

The code draft is currently unpushed at local branch `experiment/kalman-innovation-gain-gru-b300`, commit `527d2e985`; the user stop precludes new work. See the [hypothesis note](../ideas/KALMAN_INNOVATION_GAIN_STATE_GRU_20261018_R1.md) and [campaign closeout](../../campaign_2026-10-18/online_features_r1/CLOSEOUT_20261018_R1.md).
