# Causal local-trend filtering as a recurrent-feature and teacher hypothesis

**Status:** source-grounded proposal plus a separately labeled synthetic NumPy
receipt. No Challenge2 fit, grade, B300 profile or gain is established.
Hypothesis `H-KALMAN-LOCAL-TREND-20261018`; experiment
`E-KALMAN-LOCAL-TREND-B300-20261018`.

## Primary literature and transfer limits

- Haarnoja, Ajay, Levine and Abbeel, [Backprop KF: Learning Discriminative
  Deterministic State Estimators](https://arxiv.org/abs/1605.07148) (2016),
  differentiates a deterministic latent state-estimator computation graph and
  relates it to recurrent models. Its synthetic tracking and KITTI results are
  not evidence about this tabular competition or the proposed feature-only
  local-trend adapter.
- Revach et al., [KalmanNet: Neural Network Aided Kalman Filtering for Partially
  Known Dynamics](https://arxiv.org/abs/2107.10043) (2021; revised 2022),
  combines state-space structure with a recurrent learned component for
  partially known dynamics. This draft uses an analytically specified scalar
  local-trend recurrence and learns only globally tied positive Q/R scalars;
  it does not reimplement KalmanNet or claim its experiments transfer.
- Shlezinger et al., [AI-Aided Kalman Filters](https://arxiv.org/abs/2410.12289)
  (2024; revised 2025), surveys combinations of learned and Kalman-style
  estimators. It supplies vocabulary and design alternatives, not a result for
  these raw channels, validation folds or target metric.

## Local predecessor artifact

The predecessor NumPy source is archived in the draft packet at
[`kalman_numpy_prototype.py`](../../experiment_drafts/kalman_teacher_gru_distill_b300/evidence/kalman_numpy_prototype.py), SHA256
`c867df3aa80375e19cf6f7354b78abf3fede7d7b66ad1fab20c3f91f8320200a`. Its
matching [`synthetic receipt`](../../experiment_drafts/kalman_teacher_gru_distill_b300/evidence/numpy_synthetic_receipt.json), SHA256
`c3f715b0742b24dd4bc7016893ed8d331ce57e79ed17884581c4e7818269aa15`, records
finite-value, causal-prefix, exact-reset, covariance PSD, batch/streaming and
scalar random-walk-KF/constant-gain-EMA checks on 20,000 synthetic rows and four
channels. That equivalence justifies a matched EMA control. The receipt's
`Q=0.02, R=0.5` arithmetic is separate from the draft's fixed `q=0.01,r=0.1`
EMA gain; both use the scalar steady-state formula.

The recorded peak RSS is 841,376 KiB, above the intended 500 MiB support-task
limit. The prototype's resource verification is unconfirmed. The NumPy code is
not the PyTorch model, no Challenge2 data or labels were read, and none of the
receipt establishes task quality, framework parity, B300 runtime or a serving
benefit.

## Proposed local test

The linked [experiment draft](../../experiment_drafts/kalman_teacher_gru_distill_b300/README.md)
implements fresh H1024×2 GRU teachers with four scope-matched arms (raw zero
padding, matched causal EMA, fixed-Q/R local trend, learned-Q/R local trend) for
both the four-channel pilot and all 112 supplied raw channels. Each arm uses 30
paired exposed fold/seed cells. A separate H256×2 student comparison replaces
one clean same-fold family source with the learned smoother teacher. An actual
8×B300 profile gates all scientific fits; runtime remains unknown. Primary
sources motivate this design but do not validate its task-specific benefit.
