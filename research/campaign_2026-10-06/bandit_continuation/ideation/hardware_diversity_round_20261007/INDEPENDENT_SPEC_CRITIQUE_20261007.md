# Independent bounded critique: hardware-diversity hypotheses

Recorded 2026-10-07. Reviewed frozen `HYPOTHESIS_CARDS.md` SHA-256
`6dabf1dc2af7d1d193fe43d5bf9676d7a7bcedf6a90def163fdf0b92d719cc8c`.
Scope: HD-1, HD-4, HD-5, HD-7, HD-8 only; spec critique, no code, fit,
score, allocation, or changes to the frozen cards. All hypotheses remain
retained with null numerical priors. These are smallest clarifications before
implementation, not stop rules.

## HD-4: EDMD coordinates may be a reparameterized ridge

EDMD's source method estimates an observable-space evolution operator from
snapshot pairs and a chosen dictionary. It does not imply that applying this
operator creates label-predictive information. If the frozen operator `K` is
full rank and the head is an unrestricted linear readout, `z' = K^T z` spans
the same predictor class as `z`: for any direct readout `w`, choose
`beta = K^-1 w`. Isotropic ridge after the transform then changes the penalty
to `w^T K^-T K^-1 w`; it is an anisotropic regularizer. An orthogonal `K`
with the same isotropic penalty is an exact change of basis and should preserve
predictions. A rank-deficient/truncated map instead imposes a supervised or
unsupervised feature restriction. Lower one-step input residual alone does
not show better target prediction.

Smallest test: first assert direct and transformed predictions match for a
full-rank orthogonal synthetic `K` and fixed ridge. For the train-only EDMD
`K`, compare (1) direct dictionary ridge, (2) transformed ridge, and (3) the
direct-coordinate ridge with the exactly equivalent `K^-T K^-1` penalty.
Freeze the penalty and rank without using fold-0 scores; report rank, singular
spectrum, conditioning and input residual as diagnostics. Retain the arm as an
input-dynamics/regularization hypothesis even if its first implementation
reduces to ridge with a different penalty.

The cited EDMD paper describes a finite Koopman approximation from snapshot
pairs and observables; applicability of its dynamical interpretation here is
an assumption, as the card already says: https://arxiv.org/abs/1408.4408.

## HD-7: distinguish Gaussian-IB projection from direct two-target ridge

The JMLR result is specifically for jointly Gaussian variables. Its solution is
a noisy linear projection onto a CCA-related eigenbasis, with the compression
parameter controlling dimension/scale. A deterministic projection/readout
that omits the noise is best described as a Gaussian-IB-derived supervised
subspace, not the complete stochastic IB encoder. The Challenge2 labels need
not be Gaussian, and a lower input variance than PCA does not itself imply a
higher official score.

The official metric uses separate clipped-label-magnitude weights for T0 and
T1. Therefore there is no single canonical row-weighted joint covariance
unless the recipe defines one; pooling target weights silently changes the
method. Smallest source-faithful test: freeze one beta/rank in advance; fit an
unweighted train-only joint `(X, [Y0,Y1])` Gaussian-IB subspace; compare it to
same-rank train-only PCA/readout and a direct two-target ridge on the identical
eight-row/random-projection input. Use the same declared per-target head loss
and fixed checkpoint rule for all three, and score with the unchanged official
evaluator. This tests whether supervised rank restriction helps beyond a
linear predictor without claiming the Gaussian model is true. If using
label-magnitude-weighted covariance instead, preregister it as a distinct
adaptation and compute separate T0/T1 covariances/maps (or state and justify an
explicit shared-weight rule); do not call that weighting part of Gaussian IB.
Do not choose beta/rank from exposed fold-0 performance.

The paper states the Gaussian scope and the CCA-related noisy projection
explicitly: https://jmlr.org/papers/v6/chechik05a.html. The proposed use on
this dataset is an extrapolation, not a result from that source.

## HD-1 and HD-5: make streaming readout statistics target-specific; time IO

A single shared weighted Gram matrix is incorrect when T0 and T1 use different
label-magnitude weights or scored-row masks. For each output `k` separately,
freeze the training target `q_ik` and weight `w_ik`; for the metric-aligned
primary variant these could be `q=clip(y,-2,2)` and `w=abs(q)`, while raw
training targets or another weight are separate, preregistered choices. With
`u_i=[feature_i;1]` for a fitted intercept, accumulate
`G_k = sum_i m_ik w_ik u_i u_i^T` and
`b_k = sum_i m_ik w_ik u_i q_ik`, plus weight/mask totals. The fit-group and
row-mask rule must be explicit, including whether warmup rows are used for
training; the evaluator's official mask is unchanged. Solve one fixed
regularized system per target. This avoids substituting a common weight or
silently treating the two targets as identically distributed.

Smallest test for both arms: on a tiny synthetic stream where T0/T1 weights
and missing masks intentionally differ, compare chunked per-target statistics
and predictions against an explicitly materialized direct weighted solve;
cover intercept/weighted-centering, chunk boundaries, row order and zero-weight
rows. Then measure the whole path on the intended device: opening/decoding the
permitted training source, feature/state generation, both Gram accumulations,
solves, prediction write and hashes. Report bytes read/written, warm versus
cold-cache condition, wall time, peak RAM/VRAM and completed sequences. The
random-feature/reservoir update may be cheap while input decoding or repeated
hashing dominates; a kernel-only throughput number is not experiments/hour.

## HD-8: preserve level, fix path scaling and test causal Chen updates

A standard path signature is translation-invariant: it records increments and
ordered iterated integrals, not the initial absolute value. As written, the
HD-8 signature-only head discards the current projected feature level. Add the
current projected endpoint `z_t` as an explicit static feature alongside the
signature, and give the matched delay control the same endpoint block. Then
the contrast tests whether level-two order/area terms add to a shared level
baseline rather than comparing a level-aware control to a level-free candidate.

Freeze the interpolation and scale of all four projected inputs from training
groups only. For piecewise-linear row increments `dx`, the truncated update
should use `S1 <- S1 + dx` and `S2 <- S2 + S1_old outer dx + 0.5 dx outer dx`
under the card's iterated-integral ordering. Specify whether the step consumes
`x_t` before predicting `y_t`; consuming current row `x_t` is causal, while
future rows must not affect earlier predictions. Include normalized within-
sequence row position as proposed; it removes the unaugmented signature's
reparameterization invariance without adding wall-clock or sequence identity.

Smallest test: compare against direct piecewise-linear integration on tiny
paths with known signed area (including two paths with the same endpoint but
different order), then test row/chunk Chen parity, causal prefix invariance,
reset between sequences, finite values and float64-reference drift on a full
20,000-row adversarial/scaled path. Profile the full feature path only after
these pass. Keep projection width/depth frozen; if drift fails, first diagnose
training-only scaling and update convention rather than tuning it to a score.

The signature source establishes iterated integrals and Chen concatenation,
and documents translation and time-reparameterization invariance; adding the
current endpoint and row-time is a task-specific design choice:
https://arxiv.org/abs/1603.03788 (2025 v2; relevant definitions and Chen
identity also appear in the 2016 v1 cited by the card).

## Retention and boundary

These clarifications do not rank or close any arm. Preserve independent model
weights, training-only transform/statistic fitting, complete-chain split
membership, the declared target treatment, actual-device full-path profiling,
and the fixed official `global_wp_clip2_v1` grader. Any exposed fold remains
development evidence; a weak standalone score may still motivate a separately
registered teacher/complementarity test when allocation permits.
