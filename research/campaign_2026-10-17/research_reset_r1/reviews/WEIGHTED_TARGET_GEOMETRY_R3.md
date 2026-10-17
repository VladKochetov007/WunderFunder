# Weighted target geometry audit — R3

## Finding

For each scored target, the direction that maximizes population weighted Pearson over causal inputs is the conditional mean under the *selected, target-magnitude-weighted* distribution. In ordinary notation this is

\[
m_k(x)=\frac{\mathbb E[M\,|Y_k|\,Y_k\mid X=x]}{\mathbb E[M\,|Y_k|\mid X=x]},\qquad Y_k=\operatorname{clip}(Y_{k,raw},-2,2),
\]

when the denominator is positive. Here `M` is the official selected-row mask. This differs in general from the plain conditional mean. The result identifies an optimal *prediction direction*, not a unique scale or a guarantee that a finite recurrent model can estimate it. The two targets have separate weights and separate optima.

## Derivation from the implemented score

For target `k`, let `P_k = clip(P_raw,k, -2, 2)`, `w_k=|Y_k|`, and `Z_k=E[M w_k]`. When `Z_k>0`, define a probability measure over rows by

\[
dQ_k = \frac{M w_k}{Z_k}\,dP.
\]

The scorer's per-target value is exactly `Corr_Qk(Y_k,P_k)`: its weighted means, covariance, and variances are ordinary means, covariance, and variances under `Q_k`. Rows outside `M` and rows with `Y_k=0` have zero `Q_k` mass.

For any predictor `f(X)` measurable from the allowed causal inputs, put `m_k(X)=E_Qk[Y_k|X]`. Then

\[
\operatorname{Cov}_{Q_k}(Y_k,f(X))=\operatorname{Cov}_{Q_k}(m_k(X),f(X)).
\]

Cauchy–Schwarz bounds the correlation by

\[
\operatorname{Corr}_{Q_k}(Y_k,f(X))\leq
\sqrt{\frac{\operatorname{Var}_{Q_k}(m_k(X))}{\operatorname{Var}_{Q_k}(Y_k)}}.
\]

When both variances are nonzero, equality holds for `f=a m_k+b` with `a>0`. Rewriting the conditional expectation under `Q_k` gives the ratio above. Since `Y_k` lies in `[-2,2]`, its weighted conditional mean also lies there, so the score's bounded prediction range contains an optimum. If its denominator is zero at an input, that input has no `Q_k` mass and its prediction does not affect this target's score. If `Z_k` or the target variance is zero globally, the implementation returns zero for that side.

The same ratio minimizes the *pointwise weighted squared-error risk* `E[M w_k (f(X)-Y_k)^2]`: differentiating conditional risk gives `f E[Mw_k|X]=E[Mw_kY_k|X]`. That is a separate, additive-risk fact. Pearson itself is a global, non-additive score, so it has a positive-affine family of optima rather than a unique pointwise Bayes action.

For `w=|Y|`, the numerator is `E[M |Y|Y | X]` (signed square), not `E[M Y | X]`. For example, at one input let `Y` be `2` or `0.2` with equal probability. The plain mean is `1.1`; the weighted mean is `(0.5·4+0.5·0.04)/(0.5·2+0.5·0.2)=1.83636…`. If a mask selects only the `Y=+1` case from an equal mixture of `Y=-2,+1`, the selected weighted mean is `+1`; dropping `M` gives `(-4+1)/(2+1)=-1`. The selected mask must remain in the conditional ratio.

With fixed target weights and a fixed mask, `Corr_Q(Y,aP+b)=Corr_Q(Y,P)` for `a>0`; a negative `a` reverses the sign. This invariance holds only when the actual evaluated values remain an affine transform. The official scorer clips predictions after the transform, so saturation turns `clip(aP+b,-2,2)` into a nonlinear map and can change the score. A raw clip count alone does not say whether it improves the score; inspect the target-weighted mass and regrade exact outputs.

The official result is `0.5*(rho_0+rho_1)`, with one pooled correlation per target over all selected rows and sequences. It does not use one joint two-output covariance. `w_0` and `w_1` generally differ. An invertible common/contrast output rotation followed by its inverse before scoring is only a reparameterization. Measuring correlation in the rotated target coordinates, applying one shared weight to both rotated components, or constraining a nonlinear common/contrast network changes the problem or model class; none is a free score improvement.

## Literature boundary

Gneiting's primary paper, [“Making and Evaluating Point Forecasts,” arXiv:0912.0902](https://arxiv.org/html/0912.0902), Theorem 2.7, establishes that multiplying a consistent scoring function by an observation-only weight makes it consistent for the original functional under the corresponding reweighted predictive distribution. Its squared-error example gives the weighted conditional-mean ratio. This supports the weighted-measure interpretation; it does **not** prove the Pearson result above. The Pearson optimum here is the Cauchy–Schwarz derivation from the implemented correlation. The paper states the theorem for distributions with densities, while these labels are discrete; the finite-row measure argument above is direct and does not rely on that density assumption.

## Code and prior-art audit

The current [scorer](../../../../src/wnn/gmetric.py) casts selected labels and predictions to float32, clips both to `[-2,2]`, sets `w=abs(clipped_y)`, accumulates centered weighted moments per target, pools across rows, and averages the two correlations. This matches the derivation, including zero-weight rows and the bounded prediction surface.

The current [joint-code head](../../../../src/wnn/outcomes.py) uses per-code moments `E[|Y_k|Y_k]` and `E[|Y_k|]` in its ratio readout. When `ce_tilt` is enabled, [batch preparation](../../../../src/wnn/training/batches.py) weights code cross-entropy by `main_valid * mean_k |Y_k|`; the decoder then divides posterior mass by the per-code average of the two `E|Y_k|` tables. This correction is exact only when the expected CE tilt within a code does not vary with `X` (and the moment tables are exact). Coarse tail codes, finite estimates, and any within-code value variation make it approximate. The resulting head is a posterior proxy with a metric-aware decoder; cross-entropy does not itself maximize official Pearson. `sequence_class(y)` is target-derived training supervision and must not be supplied as a serving feature or gate.

Prior records narrow what remains new:

| Prior | Recorded result | What it establishes |
|---|---|---|
| Earlier atom-distribution head (archived E02 source) | The 2 October source audit preserves a class-3 categorical head trained with `mean(|y0|,|y1|)`-weighted CE, decoded per target as `sum(pi*g*|g|)/sum(pi*|g|)`, then mixed with regression using the model's own class-3 probability. The same audit records an older `readout(tilted=True)` that divides posterior mass by per-code mean `|y|` before the moment ratio. | Joint distributions, magnitude-weighted CE, tilted readout, and a regression mixture are already prior art. E02 specifically flags a weighted-CE/readout mismatch in the atom-head path; see archived [E02 source excerpts](../../../../research/unified/reviews/legacy/WunderFunder_audit_evidence.md). |
| Historical `ce_joint`/tilted code head | Tracked run card `d72a4b76:research/runs/ab17_ce_tilt.md`: one seed, one epoch, `devB764` WP `.6536` (`t0=.6458`, `t1=.6615`), no fold verdict or matched control. | A real exposure, but not a controlled estimate of the head's effect. |
| E468 joint codebook | Joint tilted delta `−.000618`; newly fitted regression anchor `−.000555`. All arms shared a newly fitted trunk with CE auxiliary loss. | Negative on that recipe, but weak evidence against joint representation because its direct control also lost. The weighted-mean readout is already known mathematics, not a new estimator. See [review](../../../../research/ELEPHANTS_REVIEW_2026-10-05.md). |
| Histogram MSE / CE | The 17 October local audit reports strict/train282 deltas `−.0001649/−.0001488` for histogram MSE and `−.0019140/−.0010158` for histogram CE; the histogram-CE strict interval was wholly negative. Fixed 80/20 parent plus histogram-MSE was `+.000062/+.000078`, short of its interval and fixed `.0001` gate. | Distributional/binning approaches already have negative or inconclusive exposure. Do not rebrand them as a new probability head. See [LOCAL_HISTORY.md](/Users/nb/Documents/wunder/audit-2026-10-17/LOCAL_HISTORY.md:134). |
| Chunk/global correlation objectives | The same local audit records strict deltas `−.000063 / +.000076 / −.000060` for macro-WMSE / chunk-correlation / pooled-correlation, respectively, and says none won on the train282 comparison. E-291's paired pooled-memory comparison was `−.001521` (95% interval `[−.002868,−.000161]`) for two-stage and `−.000414` (`[−.001260,+.000463]`) head-only. The tested correlation losses did not clip predictions like the official scorer. | Keep the tested pooled-memory recipe rejected. A clipping-semantics check is a narrowly scoped reopening condition, not evidence for a new objective. See [LOCAL_HISTORY.md](/Users/nb/Documents/wunder/audit-2026-10-17/LOCAL_HISTORY.md:129) and [metric-objective review](../../../../research/unified/ideas/EVENT_STATE_AND_METRIC_OBJECTIVE_REFRAME_20261015_R1.md). |

These are historical development comparisons, not fresh holdout claims. The local-history table's panel definitions and intervals must be consulted before comparing across lineages.

The separate metric-aware privileged-moment proposal is also already registered: the 12 October strategy describes cross-fitted numerator and denominator projections into a causal student, denominator shrinkage, and a direct-regression fallback. It is not a new idea from this audit ([strategy lines 118–122](../../../../research/unified/reviews/CHALLENGE2_PUBLIC_PRIVATE_STRATEGY_20261012.md#metric-aware-privileged-moments)).

Likewise, the current training path already includes a four-family weighted-MSE distillation branch in [losses.py](../../../../src/wnn/training/losses.py). This review does not recast family targets or auxiliary moment projection as a new head.

## Idea disposition

### Candidate: cross-fitted scalar score-to-weighted-mean calibration

For a frozen causal per-target scalar output `S_k`, estimate `g_k(s)=E[M w_k Y_k|S_k=s]/E[M w_k|S_k=s]` from honest out-of-fold predictions and training labels. This is the weighted conditional mean among predictors that can use only that scalar, so it can correct nonlinear amplitude distortion that a global affine rescale cannot. Use a predeclared low-knot table or spline, with shrinkage toward the identity; keep the two targets separate. This is a calibration map, not a new recurrent representation or a claim about event structure.

**Novelty and evidence.** A scalar-only map is distinct in mechanism from predicting the 504 outcome codes from hidden state, but its relationship to the prior histogram-MSE arm is close enough that the exact old arm should be inspected before admitting a fit. The project evidence is negative for histogram MSE/CE; there is no result here for this exact post-calibrator. The conditional-mean identity is mathematical evidence about the estimand, not evidence that the map improves this model.

**Cheapest no-fit diagnostic.** If hash-bound, component-honest OOF predictions already exist, use them without rerunning inference. On predeclared score bins, tabulate `sum(M w_k Y_k)/sum(M w_k)` by target and compare the curve across held-out components with the identity and best positive affine direction. Do not inspect or fit a map on the grade components. If only clipped predictions were saved, saturation may have erased useful input resolution; do not infer a benefit from unseen raw values.

**Falsifier.** Defer if the weighted calibration curve is affine within component uncertainty, bin estimates are unsupported/unstable, or a separately graded component set shows no positive paired official-score effect on both the pooled score and its target breakdown. All local folds are exposed development evidence, not a new holdout.

**Cost and CPU path.** Offline estimate: low CPU, no gradient updates. Serving: two tiny lookup tables, stateless per row, so no sequence-state or reset burden. If the eventual CPU endpoint is distilled from a larger teacher, estimate or verify the map on that *student's* honest OOF outputs; a teacher calibration curve need not transfer to a student with a different output distribution. No fitting, inference, or data replay was run for this audit.

### Common/contrast network: conditional only, not admitted as a new result

An unconstrained linear `common=(p0+p1)/2`, `contrast=(p0-p1)/2` head with inverse reconstruction is exactly the original two outputs and cannot gain by rotation. A nonlinear shared-common plus contrast-residual architecture would be a genuine inductive-bias change, but target-difference residual energy alone does not show that sharing helps the separately weighted correlations. The project review already warns that the official score is nonlinear and per-target weighted; a future comparison would need a matched two-head model and exact scorer. This is a conditional architecture hypothesis, not a second geometry theorem or a free gain.

### Clipping-aware pooled loss: keep deferred

E-291 already rejected the exposed pooled-memory recipe. The existing review asks first for a source-level/synthetic check of exact prediction-clipping semantics and a summary of score-weighted clipping activity on already frozen outputs; only material activity can justify a corrected matched comparison. No new pooled-correlation loss is proposed here.

## Work boundary

This review read source and recorded history only. It ran no training, fitting, gradient update, inference, or dataset replay, and changed no model or submission artifact.
