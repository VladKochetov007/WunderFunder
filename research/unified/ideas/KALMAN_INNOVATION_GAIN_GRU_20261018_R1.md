---
kind: idea
status: code_draft_published_for_b300_review
created_utc: 2026-10-18T08:18:33Z
updated_utc: 2026-10-18T08:18:33Z
hypothesis_ids:
  - H-KALMAN-INNOVATION-GAIN-GRU-20261018-R1
experiment_ids:
  - E-KALMAN-INNOVATION-GAIN-GRU-B300-20261018-R1
evidence_paths:
  - research/experiment_drafts/kalman_innovation_gain_gru_b300/README.md
---

# Innovation-conditioned latent gain inside the GRU

## Hypothesis and mechanism

**Hypothesis:** a learned diagonal gain conditioned on a causal latent innovation can help a recurrent teacher revise its latent representation when the current observation departs from a short-memory prediction. If that correction improves the fixed-size student target, a standard GRU student may retain part of the teacher's information without carrying the observer adapter into submission inference.

For encoded input `z_t`, the adapter computes a learned measurement `o_t`, local-linear prior `m^-_t` from the previous two encoded rows, and innovation `nu_t = o_t - m^-_t`. It predicts `K_t = sigmoid(b + g([o_t, nu_t]))`, then forms `m_t = m^-_t + K_t * nu_t`. The posterior and gain-weighted innovation are projected as a residual into the ordinary fused GRU input and readout. The filter-side features are vectorized over a chunk; the GRU remains a fused recurrent operation.

This is a task-trained, KalmanNet-inspired latent observer. It has no calibrated covariance recursion, physical latent-state labels, or test-time weight updates. Calling it “online learning” would be inaccurate: only its causal recurrent state changes as observations arrive; model weights stay fixed.

## Why test it after the neighboring screens

The current local output-filter comparison lost materially, while the 39 groupwise fast/slow innovation-energy channels had deltas of `+0.00005163` and `−0.00001387` across two optimizer seeds on one exposed F0 panel. A separate fixed local-trend input replay was small and uncertain. These results make a fixed output transform or another appended statistics block a poor proxy for the question of whether a **learned gain inside the recurrent update** helps. They do not reject conditional/recurrent state estimation, and they do not predict a gain for this adapter.

Detailed external source context is in the [KalmanNet source assessment](../sources/KALMANNET_INNOVATION_GAIN_GRU_20261018_R1.md). The code and frozen B300 recipe are in [draft PR #22](https://github.com/VladKochetov007/WunderFunder/pull/22) (URL to update after creation).

## What can pay off

The teacher's score alone is not the goal. The competition payoff would be a reproducible improvement in the standard H256×2 student trained from the candidate teacher cache, with the same target scale, cache coefficients, fold, seed, and update budget as its matched control. Student serving remains a regular GRU. Any teacher quality/runtime benefit must be judged against the final student package, deterministic CPU runtime, and size limits.

The B300 scale ladder begins with H256×2 / observer width32, H512×2 / width64, then H1024×2 / width128 if the measured profile and evidence justify more capacity. The primary teacher comparison is H512×2. The larger scale is not a local result or an automatic stop gate.

## Frozen comparison and interpretation

At the primary scale, compare three scratch-trained arms in the same 30 fold/seed cells: ordinary GRU, observer with context-only gain, and observer with innovation-conditioned gain. The primary contrast is innovation gain minus context-only; candidate minus ordinary GRU is a secondary complete-architecture contrast. All arms use the same source-qualified fold, initialization seed, training rows, preprocessing, optimizer budget and fixed checkpoint. The zero-initialized observer residual branches make each observer begin as the ordinary GRU function.

Then compare three matched standard-GRU student endpoints: the current four-family cache, a cache replacing only `samefold_baseline` with the context-only teacher, and one replacing that family with the innovation-conditioned teacher. Keep the other three source teachers, epochs, coefficients, scale and student schedule identical. The per-cell clean-fold teacher registry must be supplied and verified before student fits.

The 30 cells use F0/F1 with seeds1001–1008 and F2/F3 with seeds1001–1007. These four folds have prior exposure. Report equal-fold paired means, each fold separately, fitting-seed variation, and paired source-component bootstrap intervals; do not call them a pristine holdout or public-transfer result. If an interval overlaps zero, keep the exact mechanism open when student contribution or information value supports another bounded scale/control. A concrete negative result narrows that recipe; it does not close all innovation-based representations.

## Leakage and execution constraints

The adapter reads only the current encoded row and two previous encoded rows. Observer and GRU states reset for every sequence and carry across the trainer's chunks. Labels, score masks, IDs, and future rows never enter the gain path. Training uses the canonical fold-clean pipeline and full FP32; disable TF32, autocast, and quantization for the scientific comparison. Profile actual B300 shapes before fits, assign one independent fit per GPU, and retain per-update loss/LR/gradient norm, successful and skipped updates, validation score, phase timing, and peak memory.

The implementation handoff and 11-section answers are in the [draft packet README](../../experiment_drafts/kalman_innovation_gain_gru_b300/README.md). It contains no B300 profile, task fit, student cache, grade, submission, or numerical gain claim.
