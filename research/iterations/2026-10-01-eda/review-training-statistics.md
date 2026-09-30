# Independent implementation review: TDIAG2 and CSTAT1

Reviewer: research_story, author of the separate trade-mark study; not the author of either script reviewed here. Scope: code, estimands, pairing, uncertainty and resource structure. This review was completed before reading these scripts' numerical real-data results. No reviewed source file was edited.

## TDIAG2 — `training_gradient.py`

No blocking issue found in the inspected implementation. Parent and saved head checkpoints must have identical non-head state, including preprocessing buffers, before their cached trunk features are reused. Evaluation features are generated causally over the full sequence with dropout disabled. Comparing the two heads on the same features, rows and windows preserves their intended pairing. The main, structure and next-event gradients use the original weighted losses; private output blocks and shared head parameters are separated correctly. The auxiliary sum is formed before its cosine with the main gradient. A leave-one-chain-out case removes that chain from both relevant masks and recomputes each pooled objective, including its normalization and main-loss centering.

The sample uses chains outside each saved model's evaluation fold; these are historical training-source data, not a new independent validation experiment. Repeated chains across model/fold scopes and the three fixed windows must not be counted as independent replications. Metric-mass concentration is descriptive, not an inferential sample size. The paired seeds expose two saved training paths, not a population estimate of optimizer variability.

The result answers whether the saved main and auxiliary gradients align, and how one-chain omission or saved head stage changes that local geometry. CPU float32 gradients do not reproduce historical mixed precision, Adam state, stochastic batches or a complete optimization trajectory. A positive summed-auxiliary cosine does not disprove all benefit from disabling auxiliary loss: clipping and optimization coupling can still matter. Conversely, a negative cosine does not establish that the proposed ablation will improve the competition metric. A matched future ablation remains necessary; no mandatory negative-cosine gate is justified.

## CSTAT1 — `ensemble_statistics.py`

No blocking algebra or pairing issue found. The fixed raw member average is formed before clipping, preserving the frozen C ordering. Target-specific weighted moments use clipped targets and absolute target weights. The derivative coefficients correctly account for weighted means, covariance and prediction variance. Member `j` contributes `0.25 * raw_prediction_j` on rows strictly inside the ensemble clipping interval. The global chain derivative contributions use global centering and are checked to sum to the pooled derivative. Leave-one-chain-out derivatives recompute the remaining pooled moments rather than subtracting an independently centered chain correlation.

The weighted raw member covariance and residual covariance use consistent target weights. The law-of-total-covariance decomposition includes both within-class and between-class components and verifies reconstruction. Between-class covariance belongs to the pooled metric; its existence alone does not identify spurious signal. The sampled prediction caches lack embedded row IDs, so their positional alignment rests on frozen cache provenance plus shape/split checks, not a new identity proof. Exact baseline reproduction is a strong consistency check but cannot independently validate historical class labels.

The analytic estimand is a partial derivative at raw member scales `(1,1,1,1)`, with the other scales fixed and equal coefficients 0.25. It is not a fitted weight, a renormalized-weight sweep, an admissible step size or a predicted finite improvement. A common-scale derivative can be nonzero because clipping breaks scale invariance. At exactly `abs(C)=2`, setting the derivative to zero is a declared convention at a nondifferentiable point; the reported boundary counts must be inspected before interpreting a unique derivative. The floating-point baseline is evaluated exactly as frozen, while the analytic derivative describes its usual real-valued local extension rather than differentiation through float32 rounding.

Whole-chain bootstrap within fold preserves paired member statistics and accommodates unequal chain sizes. Its intervals are conditional on the selected fitted artifacts and do not account for training seeds, prior model selection or test shift. Positive-bootstrap fractions are descriptive, not selection-adjusted probabilities that a member deserves more weight. The source is processed in small batches; full target arrays and temporary member tensors fit the specified memory budget. Synthetic finite-difference, no-clipping scale invariance, degeneracy, boundary convention and covariance reconstruction tests directly target the important mathematics.

## Numerical review added after the completed fixed runs

The implementation review above preceded this numerical inspection. Sources are `training-gradient-result.json` and `ensemble-statistics-result.json`; no additional cache scoring, model run or coefficient search was performed for this section.

TDIAG2 completed in 78.38 seconds. All 24 saved-stage/window main–summed-auxiliary cosines are positive, ranging from 0.0432 to 0.2009. Both computed global-gradient clip factors are exactly one in every case. This broader sample therefore supports neither pervasive main/auxiliary conflict nor active clipping competition at these saved checkpoints and windows. Two late-window parent cases can become slightly negative after omitting one chain (approximately −0.0041 and −0.0058), which shows sensitivity rather than a robust contrary result. The concentration count ranges from 6.75 to 11.29 among 16 chains per scope; its descriptive purpose matters. Saved-head alignment generally increases, but two fold1 late-window changes are slightly negative. Nothing here establishes a causal effect of auxiliary loss on final score. An auxiliary-off ablation can remain a cheap empirical test, but its proposed conflict/clipping mechanism has been weakened rather than confirmed by this study.

CSTAT1 completed in 21.59 seconds and reproduced fixed C at **0.7024238010052848**, within floating-point error of the registered baseline. The pooled member-scale derivatives, in fixed member order, are:

| Member | Analytic local derivative | Conditional chain 95% interval |
|---|---:|---:|
| ft256dH | +0.00025384 | [−0.00003735, +0.00056500] |
| ft128H | −0.00027699 | [−0.00047131, −0.00008514] |
| ft128s4H | −0.00002811 | [−0.00032924, +0.00027012] |
| ftc500H | −0.00011199 | [−0.00040082, +0.00020824] |

Only the ft128H pooled interval excludes zero under this conditional exploratory bootstrap. Its point derivative is negative in both folds, while the fold0 interval crosses zero. No single-chain omission reverses its pooled sign. The ft256dH point sign is positive in both folds and after every single-chain omission, but its bootstrap intervals cross zero; stability to one omission is not the same as uncertainty under resampling. These results concern local raw scale, not whether a model should be removed or how far a weight should move.

Exact clipping-boundary counts are 7 and 14 scored target rows, respectively. The reported derivatives therefore retain the declared zero-at-kink convention; this review has not recomputed directional derivatives. Fixed C's metric-weighted tail fractions are 0.394% and 0.549%, while ftc500H has 1.623% and 1.745%. The common-scale derivative is −0.00016325, consistent with clipping making scale relevant; it supplies no tested finite rescaling gain.

The ten heaviest chains supply 18.1% and 18.7% of target weight; the largest single chain supplies approximately 2.7%. Between-class covariance accounts for about 19.7% and 23.3% of total target–prediction covariance. Those are legitimate contributions to the pooled objective, not proof of leakage or a reason to replace the score with macro-class performance. They explain why equal-class diagnostic estimates and the actual pooled metric can disagree. Conditional intervals and descriptive class decompositions do not correct earlier adaptive selection or establish transfer to the hidden test.
