# Objective and estimator factorization

This derivation uses the current `global_wp_clip2_v1` specification in `METRIC.md`,
not the historical per-sequence diagnostic. Write $Y_j=\operatorname{clip}(t_j,-2,2)$,
$Z_j=\operatorname{clip}(p_j,-2,2)$, and $S=$ need_prediction AND is_scored. Each
metric target has its **own** tilted distribution
$Q_j(dX,dY)=S|Y_j|P(dX,dY)/E_P[S|Y_j|]$. $X$ means all information available to the
predictor, including causal history; coarse diagnostic bins are not this full $X$.

## Official optimum, including clipping

Let $m_j(X)=E_{Q_j}[Y_j\mid X]=E_P[S|Y_j|Y_j\mid X]/E_P[S|Y_j|\mid X]$. Weighted
conditional centering gives

$$\operatorname{Cov}_{Q_j}(Y_j,Z_j)=\operatorname{Cov}_{Q_j}(m_j(X),Z_j).$$

Cauchy–Schwarz bounds correlation by
$\sqrt{\operatorname{Var}_{Q_j}(m_j)/\operatorname{Var}_{Q_j}(Y_j)}$. Equality holds
when the **clipped** prediction is a positive affine function of $m_j$, almost
surely on positive-weight support. $Z_j=m_j$ is feasible because a conditional
mean of clipped labels is in [-2,2]. Thus weighted squared error toward **clipped**
$Y_j$ has a WP-optimal representative under the same population, selection and
unrestricted function class. This is an exact population statement, not merely a
second-order approximation. Degenerate zero weights/variances follow the metric's
zero-score convention; there can then be no informative unique optimum.

Raw predictions are not freely affine-invariant if subsequent clipping saturates
them. Tested counterexample: targets [-2,.5,1] and predictions four times targets
lose WP; a feasible affine transform .5Y+.1 retains WP1. Two targets receive equal
metric weight; summing target-wise MSE under a shared denominator allocates more
risk to whichever target has more mass. Independent unrestricted heads still have
the same pointwise minima, but a shared capacity-constrained trunk need not.

## Training tilt, sampling and specification mismatch

The active recipe samples entire sequences without replacement, mixes TRAIN and
fold-allowed validation sequences, gives every sequence/chunk an update, and has
$V_{it}=1[t\ge99](p(X)^2+.03)\max(.25,1/(1+\mathrm{since}/32))$ on TRAIN rows.
The cached propensity and since are input-derived; TRAIN has no true score mask.
Validation-in-fit replaces propensity weight by $c\,1[t\ge99](\mathrm{scored}+.03)$
for sequences carrying masks, with one full-batch scalar $c$ matching aggregate
pre-since propensity mass. Since weights multiply after this replacement. The
scalar is batch-dependent, not a per-sequence recalibration. No class weighting,
episode weighting, row thinning or replacement sampler is active in the recorded
combo recipe. The sample deliberately does not reconstruct validation-in-fit.

For pooled training wMSE, $w_{ij}=V_i|Y_{ij}|$, the pointwise minimizer is
$E[V|Y_j|Y_j\mid X]/E[V|Y_j|\mid X]$. A positive deterministic input-only factor
$g(X)$ cancels in this ratio **only** with full $X$, unrestricted prediction and
unchanged sampling/labels. Scoring uses $S$, not $p(X)^2+.03$: replacing $S$ by a
propensity does not preserve the tilted mean without an appropriate conditional
independence/selection assumption. Row sampling density can also fail to cancel
when its selection depends on targets given $X$. Shared trunk restrictions matter:
a constant-model counterexample changes from -.785714 to -.4 after an input-only
factor [1,4,1] on outcomes [-2,.5,1]; saturated fits retain the original outcomes.
Consequently E-182's phrase “causal input, optimum unchanged” requires these
qualifications and does not describe the full chunk-normalized training process.

## Distillation and stopped-bootstrap labels

The trainer already clips labels before weighting. It uses $|Y_j|$, **not** the
magnitude of the blended target or raw unbounded labels. Teacher caches are scaled
and averaged first, then clipped: $T_t=\operatorname{clip}(\frac12(T_t^{(3)}+T_t^{(4)}))$.
Plain distillation target is $B_t=(1-\alpha)Y_t+\alpha T_t$, with $\alpha=.5$.
Its pooled optimum is

$$m_{B,j}(X)=\frac{E[V|Y_j|B_j\mid X]}{E[V|Y_j|\mid X]}.$$

For a teacher deterministic in the full causal $X$, this is
$(1-\alpha)m_{Y,j}(X)+\alpha T_j(X)$, generally **not** the WP-optimal mean of $Y$.
The synthetic teacher $T=-m_Y$ with alpha .5 makes the optimum constant0 and WP0.
A teacher reducing conditional variance while unbiased in the relevant *tilted*
conditional mean can preserve the optimum, but ordinary unbiasedness is not enough.

The combo uses `--distill-boot-h 8`. Define $A_t=1[\mathrm{next\_raw\_pair\_change}(t)>t+8,
t+8<T_{seq}]$. Then $L_t=A_tT_{t+8}+(1-A_t)Y_t$ and $B_t=.5L_t+.5T_t$. Raw labels,
not clipped labels, define episode boundaries. Both $A_t$ and future $T_{t+8}$ can
be outcome-dependent given causal $X_t$. No optional-stopping unbiasedness follows
from the constancy of the realized label until a future stopping time. In
particular alpha0 with nonzero bootstrap horizon still replaces eligible labels;
it is not a raw-label control. The width320 control has horizon0. Denominators
are identical between bootstrap/nonbootstrap targets, only numerators differ.

## Self-normalized chunk estimator

For a uniformly scheduled minibatch/chunk $C$, with both targets in one main sum,

$$W_C=\sum_{i\in C,j}V_i|Y_{ij}|,\quad
H_C(f)=\sum_{i\in C,j}V_i|Y_{ij}|(f_j(X_i)-B_{ij})^2,$$

$$\mathcal L_{SN}=E_C[H_C(f)/W_C],\qquad
\mathcal L_{pooled}=E_C[H_C(f)]/E_C[W_C].$$

All these data/teacher weights are detached from student differentiation. For a
random row/target occurrence under the scheduled chunk distribution, the exact
SN conditional optimum is

$$m_{SN,j}(x)=\frac{E[V_i|Y_{ij}|B_{ij}/W_C\mid X_i=x]}
{E[V_i|Y_{ij}|/W_C\mid X_i=x]}.$$

Other labels in the same sequence/batch, the other target, warmup position and
cluster regime enter $W_C$. In tilted conditional notation $Q_{train,j}$,

$$m_{SN,j}(x)-m_{pooled,j}(x)=
\frac{\operatorname{Cov}_{Q_{train,j}}(B_j,1/W_C\mid X_i=x)}
{E_{Q_{train,j}}[1/W_C\mid X_i=x]}.$$

Covariance of $|Y|Y$ with inverse mass under *unweighted* sampling alone is not
the bias: denominator tilting and the conditioning must be specified. Even iid
one-row chunks cancel |Y| exactly. With strong within-cluster dependence, a large
row count does not imply a large effective sample size or a small ratio bias.
There is no O(1/raw-row-count) guarantee asserted here.

Positive clustered example: each cluster has16 identical outcomes and one of
three observed input states. Outcomes by state are [-2,.2,.2], [-.2,1,1],
[-.7,-.7,.8], equiprobably. $W_C=16|Y|$, so SN fits ordinary conditional means
[-.533333,.6,-.2]. Pooled fits tilted means [-1.633333,.890909,-.154545]. Official
population WP is .797979 vs .843079. These fits are not global affine equivalents.
Negative examples: varying but deterministic outcome given full $X$ causes large
mass CV with no conditional-minimum change; constant mass makes the estimators
identical. High CV is a screening gate, not proof of target bias or utility.

## The actual objective has four normalizers

Recorded combo has cached distillation, EMA consistency coefficient1, structural
CE coefficient.3 and next-change auxiliary coefficient.3. Ignoring inactive
heads, each scheduled chunk uses

$$L_C=H_C/W_C+\sum_{i,j}V_i(f_{ij}-U_{ij})^2/R_C
+.3\sum_iV_i(CE_{spread,i}+CE_{branch,i}+CE_{class,i})/R_C
+.3\sum_iV_iK_i(MSE_{next,i}+BCE_{up,i})/N_C,$$

where $R_C=\sum_iV_i$, $K_i$ marks a future **target-difference** change, and
$N_C=\sum_iV_iK_i$. EMA consistency broadcasts row weights over two outputs but
its denominator is $R_C$, not $2R_C$. `--ema-metric-weight` is false. $U$ is the
lagged EMA-model output and is not clipped by this branch. Structural labels
use targets and market class; next-change labels use future labels. Auxiliary
heads share the trunk. A frozen-$U$, unrestricted-head pooled optimum combines
main coefficients $E[V|Y_j|\mid X]/D$ and EMA coefficients $E[V\mid X]/E[R_C]$;
it is a coefficient-weighted blend of tilted distilled target and EMA target.
In the true training process $U$ changes after updates, so this frozen-teacher
optimum is only an instantaneous diagnostic; it is not a stationary objective
for the whole coupled trajectory. Auxiliary effects cannot be reduced to a scalar
shift of the target-output minimizer with a capacity-constrained shared trunk.

`norm_ema` is initialized once and updated ONLY in an `elif` below the cached-
teacher and TD branches. Thus `--loss-norm running` is numerically a no-op for
both these distilled recipes. The AST-executed test confirms exact loss equality
and unchanged norm_ema with cached teachers; the nondistilled branch updates it.
Changing this flag would not test Card6 here. E-180's nondistilled null must be
acknowledged, but it does not test this bypassed branch or auxiliary/main ratios.

## What “matched update-preserving” can honestly mean

The proposed estimator contrast preserves sample membership/order,32 sequences
per batch,1000-row TBPTT chunks, reset/detach boundaries, number of updates, teacher
caches/alpha/bootstrap, EMA/auxiliary terms, clipping, optimizer and lr/beta/EMA
schedules. Set a fixed TRAIN-estimated $D=E[W_C]$ and change main $H/W$ to $H/D$;
leave the other terms unchanged. This is $s_C H/W$ with $s_C=W_C/D$.

An alignment sham uses exactly the same epoch multiplier multiset $\{s_C\}$,
permuted by a prespecified seed across chunk slots: $s_{\pi(C)}H_C/W_C$, with all
other terms unchanged. It matches marginal gradient multiplier distribution and
update count while breaking own-chunk mass alignment. It is not exact equality
of gradient norms (loss gradients and optimizer history covary). Report actual
pre/post-clip norms and realized update lengths rather than claiming equality.

Matching every individual main-only gradient norm would cancel the scientific
intervention completely: main pooled and SN gradients are collinear in a single
chunk. Likewise preserving every parameter update exactly cannot change the
trajectory. The useful distinction is correlated estimator reweighting vs an
independent scale-distribution sham, not optimizer microtuning. A second,
input-predicted-mass sham can test whether input-composition variation accounts
for the effect, but this is a separate estimand requiring preregistration.
