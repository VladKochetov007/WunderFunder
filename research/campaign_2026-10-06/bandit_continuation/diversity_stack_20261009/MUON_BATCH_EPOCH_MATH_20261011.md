# Muon batch/LR dependence and renewed epoch-scaling diagnosis

User request, 2026-10-11: inspect existing experiments and actual code, search primary literature, establish mathematical background and identify ways longer training could work. Root owns synthesis; Luna/max independently audited the frozen recipes and mathematical claims. This is analysis and proposed bounded research, not a new fit admission. Healthy local/Kaggle/VastB jobs and existing parallel directions continue. HPO stays disabled.

## What we actually tested

All scores below are official, repeatedly exposed fold1 values. These endpoints are not public forecasts or fresh confirmation. Distinct fitting seeds, evaluation component resampling and transfer uncertainty are separate.

| Historical h256 Muon-rnn hybrid, three epochs | Sequence batch | Peak LR | Score | Evidence |
|---|---:|---:|---:|---|
| Reference | 32 | 0.001 | about 0.6660 | existing three-seed reference mean 0.66591 |
| Smaller batch, unchanged LR | 16 | 0.001 | 0.66583 | scaling deep dive |
| Smaller batch, approximate square-root LR | 16 | 0.0007 | 0.66581 | LOOP_LOG pass containing 2026-10-11 line484 |
| Smaller batch, linear LR | 16 | 0.0005 | 0.66465 | LOOP_LOG line469 |
| Larger batch, approximate square-root LR | 64 | 0.0014 | unavailable | configured; no completed prediction sidecar in current remote check |
| Larger batch, linear LR | 64 | 0.002 | unavailable | configured; no completed prediction sidecar in current remote check |

Each smaller-batch arm is one seed. The scaled arms use `--optim muon --muon-scope rnn`, with AdamW fallback for other parameters. They are neither a pure AdamW batch sweep nor an optimizer-by-batch factorial. They scale LR without a matched momentum/epsilon/decay treatment. The square-root coefficients are rounded, not exact sqrt(0.5) and sqrt(2).

The original `SCALING_DEEP_DIVE_20261011.md` says the batch16 scaling arms were never tested; that statement is superseded by the completed log evidence above. The historical batch16 sidecars remain on Vast, but the NPYs no longer do. The attempted remote rescore failed on missing NPY, preserved in `ops_continuation_20261011/LR_BATCH_EXISTING_GRADES_R1.txt`. Do not refit to replace this storage gap or count the attempt as a scientific negative.

Today's strongest scratch recipe is different: h256 Muon-all × prop0 × four epochs, seeds3/4 0.66976/0.67195, mean0.670855, n=2 provisional; local seeds6/7 and Kaggle seed5 confirmation continue. Muon-all h384 improves three-epoch teachers, while Muon-all h256 alone was approximately null. Thus neither "Muon works better everywhere" nor "epoch scaling always fails" is supported. Four-epoch teacher degradation on other widths/folds and very long-run collapse remain real evidence for their exact treatments.

## Actual implementation and clocks

Historical scale/long packets bind `continuous_execution_E529/h230_source_v5_trace` in `vast_teachers_r1/bundle_b/BUNDLE_RECEIPT.json`. Current local lead uses `h230_source_v6_lead`. Audit these producers rather than assuming working-tree `src/wnn/optim.py` represents them: the frozen v6 optimizer accepts scope `all`, while the working-tree version currently accepts only `rnn` and `hidden`.

In frozen v6 `src/wnn/optim.py`, GRU/LSTM recurrent matrices are split by gate. Scope `all` also assigns qualifying input/head matrices to Muon; vectors/biases still use AdamW. The implementation requires an RNN core. Even the all treatment is a hybrid, not Muon on every parameter or a validated optimizer for every architecture.

For each matrix block the code computes

\[
M_t=\beta_t M_{t-1}+(1-\beta_t)G_t,\quad
C_t=\beta_t M_t+(1-\beta_t)G_t,
\]
\[
W_{t+1}=(1-\eta_t\lambda)W_t-\eta_t s\,Q_5(C_t),
\qquad s=0.2\sqrt{\max(m,n)}.
\]

`Q_5` is five quintic Newton–Schulz iterations after Frobenius normalization, BF16 on CUDA. It approximates the polar factor; it is not exact SVD. Bias/vector groups use AdamW, sharing the LR schedule and beta1 with Muon's momentum. Global norm clipping precedes orthogonalization: clipping raw gradients does not impose the same bound on the normalized matrix update.

The trainer performs one update per TBPTT update window, not one per whole sequence batch. With update_rows=chunk=1000 and sequence length20000, there are20 attempted updates per loader batch. The frozen split supplies source-qualified fit membership; CLI `--train-seqs 10607` alone does not prove actual dataset length. The live lead has10533 train +1416 fit-valid sequences, loader373 batches at batch32, hence7460 attempts/epoch. Incomplete loader batches are dropped.

The main weighted MSE is a ratio of weighted sums, not an unnormalized sum over batch size. Changing batch therefore changes noise and update count, and also changes the finite-sample random-denominator estimator. Rows within sequences and successive TBPTT windows are correlated; treating B×1000 rows as independent samples is unjustified.

`build_scheduler` in frozen `src/wnn/training_dynamics.py` sets OneCycle total to loader length ×20×epochs, 5% warmup, initial division25, final division100. It also cycles beta1 (normally .95 → .85 → .95). The attempted-update scheduler advances after GradScaler even if the optimizer update was skipped. Report actual completed updates and AMP skips separately. Changing epochs reshapes both LR and momentum during the original data passes; a six-epoch schedule is not a three-epoch schedule with three identical extra passes appended.

## What the mathematics establishes

**Adam family.** Malladi et al. derive an SDE approximation and a joint square-root scaling prescription. For B'=kB, LR becomes sqrt(k)×LR, beta_i becomes1−k(1−beta_i), and epsilon becomes epsilon/sqrt(k), subject to the approximation's regime and valid coefficients. Scaling LR alone is a partial test. These results do not establish the same prescription for a Muon/AdamW hybrid. [Primary paper](https://arxiv.org/html/2205.10287v1).

**Muon geometry.** Write the exact compact polar factor as P(G)=UVᵀ. For c>0, P(cG)=P(G). Let q=rank(G), r=min(m,n). Then

\[
\langle G,P(G)\rangle=\|G\|_*,\quad
\|P(G)\|_F^2=q,\quad
\operatorname{RMS}(\eta sP(G))=0.2\eta\sqrt{q/r}.
\]

At full rank, the gradient-update RMS is0.2η, excluding weight decay. This removes global gradient magnitude, not singular-vector information. The dimensions and gate blocking matter. Finite NS, small singular values, normalization epsilon and BF16 qualify the exact identities. The RMS convention is the Moonlight convention, and cannot silently be exchanged with Keller's aspect-ratio convention. [Muon scaling paper, AppendixA](https://arxiv.org/html/2502.16982v1).

For a full-gradient, exact-polar, no-momentum/no-decay step on an L-smooth loss, the descent lemma gives

\[
f(W-\eta sP(G))\le f(W)-\eta s\|G\|_*+
\tfrac L2\eta^2s^2q.
\]

This is a direct derivation, not a theorem about our complete GRU trainer. When the useful gradient becomes small, the negative term shrinks while the positive term need not. Nonvanishing normalized updates can chatter or wander; LR must become small, or the update/noise mechanism must change. Stochastic momentum convergence analyses similarly have an optimization term, a noise term and a step-size term. Sato et al.'s revised v5 theorem has the schematic bound O(1/(ηT)+sqrt((1−β)r1/B)+rbar·η), under smoothness, unbiased bounded-variance sampling and stated polar assumptions. More updates alone do not remove the remaining terms. The paper does not prove a universal Muon LR/batch exponent. Use the revised theorem, not the materially weaker/different v1 wording. [Theorem3.1, v5](https://arxiv.org/html/2507.01598v5).

**A conditional batch rule we can test.** This is our derivation under a frozen local gradient, fixed shape/filter, isotropic IID Gaussian noise, exact polar, and small signal-to-noise. Let G_B=g+σ0 Z/sqrt(B). Xie's local response is E[P(G_B)]≈(c/σ_B)g. With D seen sequence windows and T=D/B updates, drift per seen window scales as η/sqrt(B); zero-signal accumulated variance scales as η²/B for a fixed momentum filter. Therefore η∝sqrt(B) approximately preserves both. In a high-SNR, well-conditioned regime with stable polar direction, drift instead scales as η/B, motivating linear scaling. Neither conclusion is universal; rank changes, anisotropy, correlated windows and momentum transformations can break it. [Local response and explicit limitations](https://arxiv.org/html/2609.32861v1).

For a stationary plain EMA driven by IID noise covarianceΣ/B,

\[
\operatorname{Var}(M)=\frac{1-\beta}{1+\beta}\frac{\Sigma}{B}.
\]

Our practical Nesterov filter instead has h0=1−β² and h_j=(1−β)β^(j+1), j≥1. Its variance coefficient is

\[
V_N(\beta)=\sum_jh_j^2
=\frac{(1-\beta)(1+2\beta-2\beta^3)}{1+\beta}.
\]

These formulas require stationarity and independence; cycling beta violates the stationary assumption. To preserve a momentum horizon measured in seen sequence windows when B'=kB, β'=β^k matches exponential retention. That is a different invariant from holding beta fixed, so the fixed-filter LR derivation cannot be reused unchanged. Momentum before orthogonalization can improve spectral signal/noise separation under a structured gradient model, which motivates measuring spectra and testing the momentum clock. [Spectral filtering analysis](https://arxiv.org/html/2606.03899v1).

**Decay and schedule budgets.** Define S1=Ση_t and S2=Ση_t² over successful updates. The decay-only retention is exactly Π(1−η_tλ), approximately exp(−λS1) for small products. These are movement/noise proxies, not the actual net parameter displacement or a proof of generalization. At fixed data/epochs, B'=kB and η'=k^aη gives S1'≈k^(a−1)S1. Matching integrated decay then requires λ'≈k^(1−a)λ, assuming the same schedule shape and negligible skipped steps. Thus the sqrt-LR treatment with unchanged WD changes the decay dose. It should be an explicitly different treatment, not called a clean batch-rule test.

Wang/Aitchison analyze the timescale1/(ηλ), with an exact trajectory result under scale-invariant-network and initialization assumptions. Our GRU does not automatically satisfy those assumptions. Use the exact decay product as an audit quantity; do not transplant their architectural invariance theorem as a guarantee for this model. [Primary paper](https://arxiv.org/html/2405.13698v2).

## Independently checked schedule arithmetic

Luna computed PyTorch scheduler traces; root independently computed the cosine phase formula and frozen WSD multipliers and obtained matching values. Actual producer provenance/history confirms11949 fitting sequences and14920/7460 attempts per epoch for B16/B32. Configurations alone misleadingly suggest10607. No model was fitted for this calculation. The full S1/S2/phase/decay table is saved in `ops_continuation_20261011/LR_SCHEDULE_BUDGETS_R1.json`.

| Treatment, actual N=11949 | Attempts | ΣLR | Decay exponent | Decay-only retention |
|---|---:|---:|---:|---:|
| B32 E3 OneCycle, LR.001 WD.01 | 22380 | 11.216132 | .112162 | .893900 |
| B16 E3 unchanged LR.001 WD.01 | 44760 | 22.432765 | .224328 | .799053 |
| B16 E3 approximate sqrt LR.0007 WD.01 | 44760 | 15.702935 | .157030 | .854679 |
| B16 E3 linear LR.0005 WD.01 | 44760 | 11.216382 | .112164 | .893898 |
| B64 E3 approximate sqrt LR.0014 WD.01 | 11160 | 7.829893 | .078299 | .924688 |
| B64 E3 linear LR.002 WD.01 | 11160 | 11.185561 | .111856 | .894173 |
| B32 E4 OneCycle, LR.001 WD.01 | 29840 | 14.955010 | .149551 | .861095 |
| B32 E6 OneCycle, LR.001 WD.01 | 44760 | 22.432765 | .224328 | .799053 |
| B32 E15 OneCycle, LR.0005 WD.1 | 111900 | 28.041331 | 2.804186 | .060556 |
| B32 E24 WSD, AdamW LR.00015 WD.05 | 179040 | 22.639757 | 1.131992 | .322390 |
| B32 E24 WSD, Muon-all LR.0002 WD.05 | 179040 | 30.186343 | 1.509324 | .221059 |
| B32 E24 WSD, Muon-rnn LR.00025 WD.05 | 179040 | 37.732929 | 1.886658 | .151578 |

The first three epochs of the E4 OneCycle already have ΣLR14.542, versus11.216 for a complete nativeE3 schedule; E6's first three have17.980. The small-LR Muon-all24epoch WSD has2.69× the LR integral and13.46× the decay exponent of the three-epoch reference, despite a fivefold lower peak LR. These are consequential confounds. They do not prove that weight decay caused the measured collapse.

All products assume no AMP-skipped attempts. Historical B16 sqrt/linear have19/22 skips out of44760; actual applied decay is slightly smaller. Histories record timings, skip counters and evaluation withheld; they do not contain adequate epoch-averaged primary train/validation loss curves to distinguish optimization failure from generalization failure. A displayed last-batch loss is not that curve.

## Why our epoch scaling failed: evidence versus hypotheses

Measured: standard six-epoch final0.64903 after a much stronger intermediate checkpoint; h384 Muon-all e5 collapse; long scratch schedules and warp/dropout variants degraded; low-LR24epoch runs also failed; alpha.9 teacher-distilled six-epoch student0.67103 remained strong. These establish recipe-specific degradation and conditional regularization, not a universal training-information ceiling.

Several mechanisms remain distinct:

1. **The trajectory changed before the added epochs.** Stretched OneCycle changes early LR, beta and integrated movement. Same-prefix branching is needed to isolate added training.
2. **The low-LR long tests did not hold decay/movement budgets fixed.** The24epoch WSD commands use WD.05 versus reference.01, different LRs by optimizer, and a long stable-LR phase. The15epoch OneCycle uses WD.1. Lower peak LR does not necessarily mean lower total movement or shrinkage.
3. **Late noise may be amplified by normalization.** This is plausible mathematics, not a measured diagnosis of our recurrent gradients. No fitting-only stationary spectrum/SNR probe has yet demonstrated it here.
4. **Fitting objective and official correlation can diverge.** Chunk WMSE plus auxiliary heads, propensity treatment and clipping is not the full global clipped weighted correlation. Falling displayed batch loss does not establish improving evaluation WP or even falling epoch-averaged primary loss. Separate main, auxiliary and class/channel diagnostics.
5. **Repeated data and limited causal signal can still matter.** Stronger teacher-target regularization preserved long student quality, while generic warp/dropout did not rescue scratch finals. More clean causal information, useful OOD targets and different representations remain parallel mechanisms. None is ruled out by the scheduled long-run failures.

## Bounded next comparisons, in priority order

These are proposed descendants with no numerical gain prior and no new spending admission. Ordinary failed long recipes stay closed. Costs require measured actual-device profiles; current useful queues need not be interrupted.

| Proposal | Candidate/control and observable | Why it differs from failed arms |
|---|---|---|
| H-377: current-lead batch response | Muon-all ×prop0 h256, fixed3epoch recipe/seed/membership/chunk; B32 control versus B64 with explicitly sqrt-LR and decay-dose-matched treatment. Profile first. Compare official WP and fixed scratch/student mixture value. | Tests the actual winning interaction rather than old recurrent-only hybrid; changes declared batch/noise and decay clocks together. Additional LR-only treatment separates decay if first contrast is informative. |
| H-378: same-prefix low-movement tail | Freeze a successful own3epoch training state. Retain optimizer/RNG/normalization/ancestry. Compare no extra fit, a low-LR3epoch tail (initial candidate peak3e-5, no restart), and a tail with a different momentum clock, saving fixed snapshots. | All branches start at identical trained weights; no stretched early cycle or simultaneous WD×5 change. Full-state continuation requires validated scheduler handoff, not merely weight initialization. |
| H-379: noise-aware optimizer finish | First measure fitting-only fixed-checkpoint gradient spectra/SNR, direction stability and update-to-weight RMS. If late noise dominates, compare continued Muon with response-calibrated momentum-SGD using the same EMA state; matched-update AdamW is a distinct fallback, not an equivalent theorem. | Changes the late orthogonalization mechanism; the paper motivates a switch under a local surrogate but does not validate a practical GRU switch rule. |
| H-380: auxiliary-objective conflict | At the same own prefix, control retains current auxiliary coefficients; candidate decays auxiliaries while preserving primary target treatment, LR/WD and fit membership. Measure main/aux gradient alignment and fixed official endpoint. | Tests objective drift under many passes, rather than generic dropout or a lower peak LR. |

For H-377, do not silently combine every correction. First retain the historical LR-only batch treatment as an explicitly named control, then add the decay-dose correction as its own interaction. Momentum-horizon correction is another arm; disabling cyclic momentum or implementing a frozen pointwise beta schedule is needed to keep OneCycle from overwriting it. Freeze seen-sequence exposure and plot trajectories against attempted/completed updates and seen sequences. Keep the AdamW fallback LR separate when diagnosing a Muon-group LR rule. Avoid a blind factorial/HPO sweep; use the fitting-only noise probe to choose the next discriminating contrast.

Each requires root-owned frozen recipe and separate review before execution. One-seed screens can identify informative contrasts; continuation decisions use effect, component uncertainty, cost and training variation separately. Confirm promising treatment across distinct seeds before ranking its recipe mean. Do not require a universal interval gate, and do not convert local deltas into public numbers.

The existing three-teacher alpha ladder/parent continuation, channel+Pearson+run-level KD, OOD/path distillation, independent h384 teacher seeds and all-data finals continue in parallel. This analysis supplies new reopening mechanisms, not a blanket reopening of failed epoch-only recipes.

## Review and reproducibility

Root searched the cited primary pages on2026-10-11; paper versions are explicit. Luna/max code audit inspected frozen v5/v6 and actual packet argv; an independent Luna/max mathematical critic checked the polar identities, descent inequality, conditional batch derivation and Nesterov kernel. No fits were launched and no frozen producer was edited. Actual long-run source and configured-versus-observed dataset length must accompany any numeric schedule table. Remote metric SHA matches local `src/wnn/gmetric.py`.

Scheduled-check grades and complete/hash-bound predictions live in `ops_continuation_20261011/NEW_RESULT_*_R4.*`. Independent ancestry review supports the channel-KD matched comparison. Remote teachers remain supported by producer checkpoint hashes and clean-fold closure, rather than an independent binary rehash. The homogeneous target consumed here is three distinct strong seeds; duplicated checkpoint entries in unused four-family variants must not be advertised as four independent families.
