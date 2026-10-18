---
kind: source
status: reviewed_for_transfer_hypothesis
created_utc: 2026-10-18T08:18:33Z
updated_utc: 2026-10-18T08:18:33Z
hypothesis_ids:
  - H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1
experiment_ids:
  - E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1
evidence_paths:
  - research/experiment_drafts/kalman_innovation_gain_gru_b300/README.md
---

# KalmanNet-inspired innovation-conditioned GRU: source assessment

## Primary sources

- [KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics](https://arxiv.org/abs/2107.10043) replaces the analytic Kalman-gain computation with a recurrent gain estimator driven by state-estimation errors and measurement innovations. The paper retains a predictor/corrector structure. Its reported results are for state-estimation benchmarks with an explicit state-space interpretation.
- [Recursive KalmanNet](https://arxiv.org/abs/2506.11639) explores recursive neural implementations of state estimation and Kalman-like updates. It is a recent preprint, and its reported task and labels are not this competition's anonymous market-state regression problem.
- [KARNet](https://arxiv.org/abs/2305.14644) is a neighboring learned filtering proposal; it is not evidence that a Kalman representation transfers to this dataset.

The papers provide an operator-level inspiration: estimate a correction gain from innovations while preserving a causal state update. They do not show that the latent channels in this competition are physical states, that a Bayesian covariance is identifiable here, or that a learned gain improves weighted Pearson correlation.

## Transfer to this task

The proposed adapter uses a learned latent measurement embedding, a two-row local-linear prior, an innovation, and a bounded diagonal gain. It has no observed state labels, explicit process/measurement covariance, or likelihood term. It should be described as a **KalmanNet-inspired latent observer**, not a calibrated Kalman filter. The end-to-end task objective learns its latent coordinates and gain.

The motivation is local and limited. On repeatedly inspected F0 data, fixed output filtering hurt, groupwise fast/slow innovation states were near-neutral and seed-sensitive, and fixed local-trend input replacement produced a small uncertain positive screen. Those observations argue for testing an innovation-conditioned update inside a trainable recurrent teacher; they do not support an expected gain.

The adapter's `context_only` control preserves the same parameter count, state path, zero-initialized residual branches, and training schedule while withholding innovation from the gain network. A plain GRU at the same width is the capacity/reference arm. The strongest alternative explanation is that the observer is redundant with the GRU's native reset/update gates or simply adds a useful residual branch.

## Limits and next evidence

The local observer state resets per sequence and carries through chunks; it never updates weights at inference and never reads labels. Therefore this tests causal recurrent state adaptation, not test-time training or online parameter learning. Before any B300 fit, verify chunk/full-sequence parity, prefix invariance, reset behavior, strict-FP32 execution, source identity, and full-shape memory/throughput on the assigned hardware. Then run the frozen matched cells and grade the standard-GRU student endpoint.

No outside dataset, pretrained weights, numerical result, or performance claim is imported from these papers.
