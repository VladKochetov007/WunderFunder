# Blind method proposal: causal target logic and compact heads

**Pre-outcome timestamp:** 2026-10-04 22:46:05 UTC. This memo was committed to
the campaign record before inspecting any Sol EDA output. It is a proposal, not
experimental evidence. No data-heavy read, fit, GPU work, or deployment build
was performed for this memo.

## Metric target and target-law facts

For each target, the scored value is (Y=\operatorname{clip}(T,-2,2)), its
weight is (w=|Y|), and the score is a Pearson correlation under the measure
weighted by (w). Define the tilted distribution

\[
 Q(dx,dy)=\frac{w(y)P(dx,dy)}{\mathbb E_P[w]}.
\]

Then

\[
 \mathbb E_Q[Y\mid X=x]
 =\frac{\mathbb E_P[wY\mid X=x]}{\mathbb E_P[w\mid X=x]}.
\]

For every prediction (f(X)), \(\operatorname{Cov}_Q(Y,f(X))=
\operatorname{Cov}_Q(\mathbb E_Q[Y\mid X],f(X))\). Cauchy–Schwarz therefore
shows that the maximum population weighted correlation over functions of the
available causal input is attained by an affine transform of this tilted
conditional mean; using the mean itself stays inside `[-2,2]`. If its weighted
variance is zero, the target score is zero. The reasoning applies separately
to each target; their scores are averaged equally. It does **not** say that an
unweighted conditional mean, a product of event probability and outcome
magnitude, or an inferred physical decomposition is optimal. The exact metric
clips before weighting, selects scored rows globally, and ignores zero-weight
observations ([metric specification](../../METRIC.md), implementation in
[`gmetric.py`](../../src/wnn/gmetric.py)).

Old raw TRAIN-only target-law work used component-disjoint fit and held-out
sequence panels. It found an exact event-persistence pattern on that panel:
target values stayed fixed off the feature-only `a6` event sentinel and
changed on sentinel rows, with a 99.995% event/change rate in class 3. Current
code-derived spread explained only a minority of weighted variance; the
structural oracle replacing the common component with current spread gained
only 0.0005, while realistic substitutions lost. Direction and most remaining
variation were future-path dependent. A simple first adverse-cross law was
only partial and had large unknown mass, especially in classes 0 and 1. The
old report explicitly treats these as conditional panel findings, not as a
public gain or a fully identified target law
([TRAIN target-law report](../campaign_2026-10-01/data_eval/target_law_report.md)).

The useful implication is narrow: exploit the observed event/persistence
invariant as a causal transition cue, but let training learn the target pair
from sequence history. Do not hard-code the spread as the target, decode
unidentified quote columns, or assume a particular stop/exit story.

## Candidate 1: event-synchronous target-state filter

Keep the current causal sequence encoder. Let its hidden state (h_t) consume
every supplied row through (t), so quiet-row features can still inform the
next transition. Maintain a small two-target prediction state (z_t\in\mathbb
R^2). Initialize it with a learned warm-start head after the 99-row warm-up.
At a current-row event sentinel (e_t\in\{0,1\}), update

\[
 z_t = z_{t-1} + e_t\,\Delta_\theta(h_t,z_{t-1}),
 \qquad \hat y_t=\operatorname{clip}(z_t,-2,2).
\]

The event bit is read from the current input only; it never looks ahead. The
prediction is held exactly between events. Train on all fitting scored rows
with the metric-weighted squared loss

\[
 L_{WP\text{-}mean}=\frac{\sum_{t,k}m_t |Y_{t,k}|
   (\hat y_{t,k}-Y_{t,k})^2}{\sum_{t,k}m_t |Y_{t,k}|},
 \quad Y_{t,k}=\operatorname{clip}(T_{t,k},-2,2),
\]

where (m_t) is the fitting mask. Per-target scalar normalization may be used
for numerical balance only if its fixed constants are recorded; it must not
change the intended weighted conditional-mean solution. This objective is a
direct estimator of the score-relevant tilted mean. Do not train only on event
rows: the global score weights the rows in each segment, and long segments
must retain their row exposure.

**Mechanism and risk.** A small output-state update makes the known label
piecewise-constant structure explicit and costs two stored floats and a tiny
transition head in the streaming callback. It does not need an extra RNN or a
full target-history cache. A violation of the persistence law on the real
current fit domain would make hard holding harmful; an event sentinel that is
unavailable or encoded differently in the scorer would also invalidate this
arm.

**Matched controls.** Use the same encoder, initialization, optimizer, rows,
checkpoint and head parameter count for (a) event-gated updates, (b) updates on
every row, and (c) event-gated updates with the event bit shuffled only within
training segments as a negative control. The all-row model must have the same
transition-head weights, even though its number of state updates differs.
Also retain a conventional two-output head with the same trunk and training
recipe. Compare full-row prediction outputs, not only event transitions.

**Cheap falsifier.** Before fitting, replay a small preregistered, component-
disjoint TRAIN sample with stored raw targets and verify the target-change
rate, missed-event fraction, and fraction of weighted score mass that lies in
multirow event-free segments. Report by class. Then, using synthetic sequences,
prove that the state resets at sequence changes, ignores future suffixes, and
changes only on a current event. If the event-free weighted target changes are
material, drop the hard-hold arm and keep a soft learned gate as a separately
declared model.

## Candidate 2: marginal metric-tilted finite-support heads

Attach two small conditional categorical heads to the shared causal encoder.
Each predicts a distribution (q_k(y\mid h_t)) for one clipped target over a
compact, TRAIN-defined support or ordered bins on `[-2,2]`. The distribution is
explicitly the metric-tilted law (Q_k(Y\mid X)), not the original target law.
Fit it by weighted log loss

\[
 L_{tiltNLL}=-\sum_{t,k}m_t |Y_{t,k}|
     \log q_k(Y_{t,k}\mid h_t).
\]

In the population limit the minimizer is
\(q_k(y\mid x)\propto |y|P(Y_k=y\mid X=x)\), normalized over (y); therefore
the prediction is the ordinary expectation under the learned tilted head,
\(\hat y_k=\sum_j q_{k,j}v_j\). It is **not** the ordinary target-distribution
mean and it is not computed by multiplying the original mean by a probability.
This construction is motivated by the metric derivation above. The loss
\(-w\log q\) is not a proper score for the original (P(Y\mid X)); it is
deliberately fitting the tilted (Q) that determines the weighted target
mean. This distinction is essential: weighted log scores can otherwise
miscalibrate the original forecast density ([Brehmer & Gneiting, *Weighted
scoring rules and hypothesis testing*](https://arxiv.org/abs/1611.07345)).

If targets are not exactly on a compact discrete support, use fixed ordered
bins and record the quantization bound; compare 16 and 32 bins as a declared
resolution control, or use a small smooth distribution with a deterministic
quadrature readout. Do not tune bin edges on the evaluation panel. A direct
metric-weighted MSE head is the required equal-capacity point-estimation
control. An unweighted categorical likelihood with ordinary-mean decoding is
the required readout control. A small fixed CRPS or likelihood diagnostic can
check calibration, but model selection remains on global WP.

**Mechanism and risk.** The head represents uncertainty over plausible
outcomes and computes the exact metric-relevant functional rather than asking
a point regressor to absorb every mode. The weighted categorical output is
cheap: for 16–32 support points it adds 32–64 logits per row. On roughly 37M
required rows, benchmark the complete one-CPU callback with the actual base
model before admission; logit cost and any loss of output compression must fit
the 60-minute and 20-MB limits. If weighted probabilities are poorly
calibrated under component shift, the ratio/expectation can be worse than the
direct MSE control.

**Cheap falsifier.** First use a tiny component-disjoint TRAIN sample to count
distinct clipped target values and compare in-support weighted mass for the
chosen support. On a synthetic two-point conditional law, show that the
weighted-likelihood head converges to the tilted frequencies and that its
mean equals \(E[|Y|Y\mid X]/E[|Y|\mid X]\); show that unweighted likelihood
does not. On real data, begin with one head and one component split, using the
same frozen trunk and direct-MSE control. Reject this head if binning changes
the readout by more than a preregistered tolerance or if the weighted NLL
improves while paired global WP and class guard worsen.

## Candidate 3: shared latent paired-outcome codebook

Use a shared encoder followed by a small conditional mixture over (J) paired
outcome atoms ((v_{j0},v_{j1})\in[-2,2]^2):

\[
 p(y_0,y_1\mid h_t)=\sum_{j=1}^J \pi_j(h_t)
    K_0(y_0\mid v_{j0})K_1(y_1\mid v_{j1}).
\]

The atoms may be bounded learned pairs or a TRAIN-fitted target-value
codebook. They are predictive basis vectors, not claimed economic labels
("spread", "winner", "loser", and so on). Train joint log likelihood plus a
small, fixed metric-weighted mean loss on the model's decoded predictions;
the likelihood captures paired regimes, and the direct term aligns the
one-dimensional scoring target. A simpler first version uses a discrete
codebook and cross-entropy on the pair ID; if targets are continuous, define a
smooth fixed kernel or fixed bin assignment before held-out scoring.

For a model of the original joint law, decode each target with its own exact
metric tilt:

\[
 \hat y_k =
 \frac{\mathbb E_p[|Y_k|Y_k\mid h_t]}
      {\mathbb E_p[|Y_k|\mid h_t]}.
\]

The two denominators differ; a single shared mixture probability multiplied by
one inferred magnitude is not valid. An alternative is to parameterize two
tilted marginals directly, but then this is no longer a normalized joint
forecast. If event persistence is retained, hold the mixture logits within an
event-free segment and update on the current event exactly as in Candidate 1;
use a no-hold version as a control so that any gain can be attributed to
paired outcome structure.

**Mechanism and risk.** A shared latent component can express that two target
outcomes co-occur in a small number of future regimes, while a pair-valued
codebook shares statistical strength between margins. It does not require
physical columns to be identifiable. Its main risk is atom collapse or an
unsupported low-cardinality law; a paired NLL gain can also fail to improve
the two marginal tilted means. With small (J\) the output is compact and the
inference work is a small weighted sum, but the full runtime/package check is
still binding.

**Matched controls.** At equal shared-trunk and output-parameter count compare
(a) shared paired codebook, (b) two independent per-target codebooks, and
(c) two direct weighted-MSE scalar heads. Hold fixed split, training rows,
seed, optimizer, schedule, teachers, checkpoint and event/no-event exposure.
Do not compare an event-held paired mixture only against an unheld scalar
head.

**Cheap falsifier.** On synthetic correlated and independent two-target laws,
confirm exact marginal tilted readout from the mixture and test that only the
correlated setting benefits from shared atoms. On a small component-held-out
TRAIN panel, compare event-conditioned joint NLL, per-target weighted mean
calibration, global WP and runtime. If the paired likelihood improves but
neither weighted mean improves, the codebook is not useful for this metric.

## First experiments and decision order

1. Freeze the base checkpoint and evaluation row set; collect only a small
   component-disjoint TRAIN diagnostic to validate current event-law rates and
   target-support cardinality. This is a cheap design check, not an outcome
   search.
2. Implement synthetic-only scorer/readout checks for the tilted-mean identity
   and for candidate state reset, current-row causality, and exact event hold.
   These deterministic invariants can be checked without GPU.
3. If the real event invariant remains strong, run one matched event-gated
   versus all-row-head comparison. Keep a direct scalar weighted-MSE control.
   Only then test the tilted marginal head on the same split and frozen trunk.
4. Try the shared paired codebook only if the marginal-head pilot shows stable
   outcome support and the two targets retain useful cross-dependence after
   conditioning on the frozen representation. Do not make it the first model
   change.
5. Promote only a fixed recipe with positive paired component evidence on more
   than one partition or seed, no per-class harm beyond the predeclared guard,
   no leakage, and measured one-CPU full-test runtime <=52 minutes as the
   operating target (60 minutes hard) with ZIP <=20 MB. Calibration, NLL,
   target-law exactness and class-wise correlations are diagnostics; none is a
   substitute for the final global WP comparison.

All three proposals preserve the callback contract: current row and causal
past only, state reset at each sequence boundary, finite two-value output on
every required step, fixed model membership, and no inference-time target or
mask access. Existing public seed-4 score 0.6660 is a deployment anchor, not a
validation estimate of any proposed model.

## Literature notes

- Mei and Eisner's Neural Hawkes Process lets prior event history change
  subsequent event-type intensities, illustrating event-conditioned causal
  state. Its continuous-time point-process assumptions are not evidence that
  the present labels follow a Hawkes law
  ([primary paper](https://arxiv.org/abs/1612.09328)).
- Wang et al.'s recurrent marked temporal model treats next-event time and
  mark jointly, which motivates pairing transition/event information instead
  of fitting them independently. It concerns event prediction, not this
  challenge's clipped weighted-correlation target
  ([primary paper](https://arxiv.org/abs/1701.03918)).
- Conditional sum-product networks show how a neural conditional density can
  retain tractable multivariate inference; this supports the compact paired
  codebook idea but does not validate its inductive bias here
  ([PMLR paper](https://proceedings.mlr.press/v138/shao20a.html)).
- Strictly proper scores identify a predictive distribution under their
  stated target law; weighted likelihood requires care about which tilted
  distribution is being estimated
  ([Gneiting & Raftery](https://doi.org/10.1198/016214506000001437),
  [Brehmer & Gneiting](https://arxiv.org/abs/1611.07345)).
