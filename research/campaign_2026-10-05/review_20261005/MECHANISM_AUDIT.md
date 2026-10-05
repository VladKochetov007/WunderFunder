# Second pair of eyes: mechanism audit and discriminating tests

**Scope.** Read-only audit of the October 5 target-mechanics evidence, E468's
joint-target heads, E467/E469's sequence-component screen, and selected October
3 EDA results. No data loader, scoring run, training, or provider work was
started. This report uses the raw saved reports/results where available; it does
not treat the prior idea index or an agent interpretation as evidence.

The current contract names `global_wp_clip2_v1`, a fixed public anchor of
0.6660, and a new honest-expected milestone of 0.7000. The current 2,274-file
index and [SECOND_PAIR_OF_EYES.md](../../SECOND_PAIR_OF_EYES.md) predate the
resumption and still describe the old paused/0.6641 state. They are useful
navigation aids, not current status. See
[RESEARCH_CONTRACT.md](../RESEARCH_CONTRACT.md),
[RESUMPTION.json](../RESUMPTION.json), and
[REVIEW_FILE_INDEX.md](../../REVIEW_FILE_INDEX.md). The index is a snapshot of
tracked files at its named base commit and excludes some local numerical
artifacts; it is not an inventory of today's full evidence.

## What the evidence supports

### A strong recurrent predictor needs to keep updating during quiet spans

In `target_mechanics/strong_baseline`, a fixed parent/head average scores
0.774416 WP on a 16-component panel whose groups were all used for supervised
fitting. In the separately selected 16-group excluded-row panel, it scores
0.742757. Replaying only the last predictor output since an observed `i0_p0` or
`i0_p11` change lowers WP to 0.692929 and 0.666532, respectively (deltas
−0.081486 and −0.076225). This rejects that exact hard output-hold rule. The
recurrent hidden state still updated during the hold probe, so it is not a test
of a learned event-conditioned state-update cell. The excluded groups avoid
direct supervised row-group fitting overlap, but 16/16 parent and 15/16 head
components link to fitted rows; both panels are historically exposed. These
scores are structural diagnostics, not clean generalization or public forecasts.

The label/event association itself is strong but not deterministic. For
transitions indexed by the scored destination row, 3,761/3,763 fitted-panel
label changes and 3,841/3,842 excluded-panel label changes coincide with a
current observed event. Those rates motivate event-aware *state updates* or
conditional readouts; they do not license label gating or output freezing.
Separately, on stable class-3 transitions the full model's quiet-span updates
reduce weighted squared error on average for both targets, while class 0 t0
slightly worsens. A useful mechanism must allow non-event evidence to update the
state and must not assume one universal benefit.

Evidence: [strong-baseline report](../target_mechanics/strong_baseline/REPORT.md),
[fitted residual report](../target_mechanics/strong_baseline/residual_report.json),
[excluded residual report](../target_mechanics/strong_baseline_excluded/residual_report.json),
and the destination-mask event audit summarized in the report.

### The principal residual is relational, but rotation alone is not a model

After clipping, 84.3% of fitted-panel and 85.4% of excluded-panel joint
sum/difference residual energy lies in the difference mode; class 3 is about
92.0% and 91.1%. This is an error-energy decomposition, not additive WP
evidence, and target variance is itself mostly in the difference mode. On class
3, wrong-sign errors contribute about 46–50% of weighted squared error even
though weighted sign accuracy is often high. Both direction and amplitude
remain relevant. A free sum/difference rotation is an invertible coordinate
change; useful capacity must come from conditional interactions, dynamics, or
regularization, not the basis name.

The fixed head improves WP over the parent while its weighted MSE is worse in
both targets and both panels. Prediction dispersion changes too. This indicates
that optimizing a pointwise weighted-MSE surrogate does not capture every
finite-model effect on the clipped, globally centered metric. It does not prove
that a more complicated distributional objective will help.

### E468 rejects a narrow frozen-state codebook head, not joint dynamics

E468 uses two historical GRU128 members, their fixed fold-0 state export,
1,405 fit groups and 468 evaluation groups (141 components, 808,863 scored
rows). The parent pair scores 0.677813. All five anchored head arms are below
it: joint-tilted −0.000618 (component CI [−0.001249, +0.000047]), ordinary
joint −0.000584, shuffled tilted −0.000550, independent marginals −0.001354,
and regression anchor −0.000555. Joint ordinary versus joint tilted is only
+0.000034 [−0.000040, +0.000094]; the independent-marginal control is clearly
worse than either shared-code arm. These comparisons argue against promoting
this 32-pair-atom/16-per-target, one-pass frozen-state head.

They do **not** test the current public model, an end-to-end learned recurrent
representation, continuous residuals around support atoms, a higher-resolution
or adaptive support, or a posterior state that evolves with observed events.
The E2E wrapper in the package was preflighted but not part of the scored E468
result. Quantized labels must be assigned from clipped targets: the original
frozen-state code had a raw-versus-clipped mismatch affecting 6.35% of fit rows,
which was corrected before the scored run. Do not revive that old source.

Evidence: [E468 result](../joint_target_model/RESULT.json),
[report](../joint_target_model/REPORT.md),
[independent review](../joint_target_model/INDEPENDENT_REVIEW.md), and
[preregistration](../joint_target_model/ROOT_PREREG.md). This historical fold-0
parent is not the October 5 24/16 cohort or the current public model.

### E467/E469 provide a small-component lead, not a strong-model win

The 30 fits use 20 features, width 32, a single fixed epoch/474 updates and the
same repeatedly exposed 24-fit/16-test cohort. On the diagonal memory, learning
stable decays over the fixed spectrum improves WP by +0.003630, +0.003816 and
+0.003043 across the three specified fit seeds; all three component-bootstrap
intervals are positive. That is the cleanest repeated within-recipe signal.
Yet the learned-SSM/MLP mean is 0.4741 versus 0.5050 for the small GRU/MLP, so
this is not evidence to replace the strong GRU with a diagonal SSM.

The spline-head gain over MLP on learned SSM narrows from +0.038616 to
+0.028818 to +0.012520; the last interval includes zero. The small GRU spline
effect changes from +0.020008 and +0.018779 to −0.003533. Attention contrasts
also vary by seed and base memory. The spline is seven-knot piecewise-linear
and KAN-inspired, not the original cubic KAN. The SSM is a diagonal,
input-independent decay model, not a selective SSM; the attention uses a small
positive-kernel fast-weight memory. None tests these components as a
zero-initialized residual branch on the production-width, two-layer incumbent.

All CIs are conditional on the same 16 exposed test components. The variation
between fitted seeds is visible, but three fixed initializations do not estimate
a general training-seed distribution. The carried-state TBPTT recipe also
updates weights between chunks while carrying a state computed under older
weights. The exact components and limitations are in
[three-seed report](../sequence_components/THREE_SEED_REPORT.md),
[E467 preregistration](../sequence_components/PREREG.md), and
[E469 preregistration](../sequence_components/SEED_REPLICATION_PREREG.md).

### October 3 negatives are narrower than their labels suggest

- **Clock family:** the October 3 own-clock model beat shared clocks in pooled
  ridge WP but failed the frozen half-split rule (+0.0361 / −0.0011); the
  follow-up event-vs-row result was −0.000544, with t0 −0.007712 and t1
  +0.006624. Post-score target-class effects reverse in the largest class.
  E.g. `signal`'s own-vs-shared mean deltas were positive in three target-derived
  classes but −0.057875 in class .6569. This kills a universal clock claim, not
  a feature-only target-specific clock adapter.
- **Lag/phase family:** dense EMA beat phase-8/phase-7 and cycle-8 controls in
  the linear report; `puzzle_b`'s lag-4 effect was +0.007909 but fell to
  +0.000613 after removing two post-selected sequences. Do not reopen a period
  scan. A predeclared lag/innovation feature is alive only if it beats an
  equal-count shifted-lag sham on new components and then improves the strong
  model.
- **Market decoding:** HFT's physical spread/quote identity failed (negative
  implied spreads and poor exact state decoding), but joint target-jump/event
  association remained strong. The report explicitly disclaims identification
  of instrument, tick, or horizon. Retain event information without importing
  the physical story.
- **Static target laws:** the `statphys` report falsified a timeless exact sum
  lookup from current prices and showed path-dependent repeated-state witnesses.
  `puzzle_a`'s six-sequence subset had no label change over unchanged-i0 runs;
  the larger strong-baseline scored-destination audit found rare exceptions.
  These are different samples/masks, so treat stable runs as frequent and
  event-linked transitions as highly predictive, not as a universal law.
- **Weak-to-strong transfer:** nonlinear offsets that helped a weak ridge failed
  on the strong GRU. This rejects that transported offset recipe; it does not
  rule out a cross-fitted, zero-initialized residual adapter trained on strong
  hidden states.

Primary reports: [`signal`](../../campaign_2026-10-03/eda/signal/REPORT.txt),
[`econometrics`](../../campaign_2026-10-03/eda/econometrics/REPORT.txt),
[`linear`](../../campaign_2026-10-03/eda/linear/REPORT.txt),
[`puzzle_b`](../../campaign_2026-10-03/eda/puzzle_b/REPORT.txt),
[`HFT`](../../campaign_2026-10-03/eda/hft/REPORT.txt),
[`statphys`](../../campaign_2026-10-03/eda/statphys/REPORT.txt), and
[`nonlinear`](../../campaign_2026-10-03/eda/nonlinear/REPORT.txt).

## Five discriminating tests on the strong predictor

All candidates should start from the exact current two-layer parent and run on
the same frozen fit/exposed-test components, with an unmodified baseline arm,
identical fit order/seed, a prespecified epoch/update count, official clipped
global WP, target-wise deltas, and paired component uncertainty. Preserve all
arms. These are proposals, not evidence or approvals to launch.

| Priority and mechanism | Cheapest strong-baseline comparison | What would falsify it |
|---|---|---|
| **1. Event-conditioned latent update with a quiet-span innovation path.** Feed current event/age and a small causal feature-innovation summary into a zero-initialized low-rank update adapter. Keep recurrent updates available on non-event rows; do not freeze outputs or state. Event coincidences motivate when to adapt, while hard hold shows updates matter between events. | First head-only probe on saved strong hidden states with event age, recent event embedding and current-minus-running-input summary. If positive, one full-model continuation. Controls: same-size row-age/innovation adapter and within-sequence frequency-preserving shuffled event times. | Event arm fails to beat baseline and both controls on component-paired global WP, or its benefit vanishes on the second preregistered component block. Any use of target changes, future events, or label-class gates disqualifies it. |
| **2. Train against the actual clipped weighted-Pearson objective.** Weighted MSE has the right conditional weighted-mean target but is a surrogate for a centered, scale-invariant finite-sample score. A differentiable pooled-correlation loss could change optimization of dispersion/covariance that the residual report shows matters. | Same current model/initial bytes and minibatch order; compare existing weighted-MSE recipe to exact per-target weighted Pearson over complete sampled sequences, with one explicitly fixed stabilizing denominator. Add a hybrid WP+small WMSE anchor only as a separately named arm. Inspect gradient norms and clipping incidence before the fit. | No component-paired gain across a second fit seed, instability/saturated clipping, or improvement confined to training groups. Do not claim a new Bayes estimand: unconstrained weighted MSE and weighted Pearson share an affine-optimal conditional predictor under ideal assumptions. |
| **3. Target-asymmetric temporal adapters.** t1 is harder and older own/event-clock tests disagree by target. Share the recurrent trunk but let t0 and t1 read row-clock versus observed-event/innovation features through separate small adapters. | Cheapest: fit-only adapters on frozen strong states with (a) both clocks to both heads, (b) target-specific clocks, (c) shared clock duplicated to both heads; match parameter count. Gate only from current/past raw input. | The targeted assignment does not beat both-clock and duplicated-shared controls on component paired WP, or the gain is only the largest target-derived class. This closes the tested adapter assignment, not clocks generally. |
| **4. Joint support plus continuous escape residual.** The joint head result leaves room for a structured paired outcome bias only if it improves representation/regularization. Use train-only clipped paired atoms plus a continuous residual, initialized exactly at the current prediction. | Head-only strong-state comparison: direct head; joint atom probabilities plus zero-init residual; independent marginal atoms plus matched residual; shuffled paired atoms. Keep codebook fit on training components only. Report support-escape frequency and t0/t1 separately. | No gain over direct and independent controls on new components, large escape dependence on test amplitudes, or success only without correct clipping. The old coarse frozen head itself stays a negative. |
| **5. Additive nonlinear residual readout on the actual hidden state.** E467's spline signal is too small/seed-sensitive to promote but sufficient to test cheaply at the scale where the incumbent has useful features. | Train a zero-init piecewise-linear spline residual on saved strong hidden states; match parameters with a hinge/ReLU MLP residual and the no-residual head. Use fixed knot locations derived only from fit hidden activations; no test occupancy tuning. | Spline does not beat the matched MLP or direct head on a new component block, or the result disappears under a fixed second fit seed. This tests local edge functions on strong representations, not “KAN” as a whole or full-model replacement. |

For each head-only screen, compare full validation component IDs and sequence
rows rather than reusing the historical E468 fold or treating class-balanced
16-component panels as representative public data. A screen can reject a
mechanism; positive screens need a new component block and another fit seed.
No sum of these small effects can be translated into expected public 0.7000
without a calibrated public-transfer model.

## Current external-method context

Recent work does not supply a ready-made winner for this setting. [GAS-Norm
(CIKM 2024)](https://arxiv.org/abs/2410.03935) uses score-driven input mean/variance
filters with a forecasting network; its reported data are Monash series, so it
motivates a narrowly tested causal normalization path, not a performance claim
for anonymized 20,000-row sequences. [Adaptive Markovian variance switching
(2024)](https://arxiv.org/abs/2402.14684) makes online latent variance adaptation
a principled possibility, but its examples are synthetic and electricity-load
forecasting. [CoRe (September 2026 preprint)](https://arxiv.org/abs/2609.19670)
adds frequency-coherence and cross-variable relational losses for direct
multi-horizon forecasting; this challenge emits two contemporaneous
per-row indicators, so the paper's horizon losses do not transfer literally.
The older high-frequency TCN paper's discrete change categories are likewise a
domain analogy, not challenge-specific validation ([Dai et al.](https://arxiv.org/abs/2107.00261)).

These sources reinforce two practical directions—nonstationary causal state
and objective/target-structure alignment—while leaving all local falsifiers
necessary. None establishes current benchmark SOTA on this task.
