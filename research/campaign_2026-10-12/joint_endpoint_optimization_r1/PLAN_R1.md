# Joint optimization of model selection, precision and blending

Root plan, 2026-10-12. User explicitly requests maximal submission quality under
the competition CPU deadline, considering many seeds/families, quantization,
mixed precision, constrained optimization and stacking. This extends the existing
blend optimization and compression notes; it does not reactivate HPO or authorize
new rentals, top-ups, billing changes or teammate weights/code.

Status: literature and equations verified; bounded numerical implementation
completed and integrated with eight synthetic checks passing; real-data selection pending. No new optimized weights, low-bit runtime advantage,
cross-validation gain or globally optimal submission has been established.
Existing frozen ZIPs remain unchanged. Matched F0 student delivery continues.

## Optimize the complete delivered endpoint

An option is an independently trained recipe/seed/checkpoint plus a concrete
precision implementation. Select a subset S, precision q for each member or
matrix group, and fixed coefficients a. Maximize the official two-target
`global_wp_clip2_v1` of the complete served mixture, subject to:

- actual ZIP <=20,000,000 bytes;
- offline Python3.11, one CPU, <=16GB scorer RAM, total test execution <=3600s;
- causal processing of every row, deterministic predictions and sequence reset;
- declared independent weights and clean teacher/initializer fitting ancestry.

Selection uses recipe-matched CV siblings, not fitted final-weight validation
scores or leaderboard-derived weights. When complete compatible folds exist,
pool their source-qualified OOF predictions using the official metric, and also
report every paired within-fold delta and distinct-seed mean. Until then, compare
only the eligible matched folds; do not invent missing-fold results or average
the absolute scores of unrelated F0/F1 endpoints into a ranking.

Cost is not simply model count or bit count: shared trunks, readouts, feature
construction, cache residency, ISA dispatch and ZIP compression matter. Measure
the actual whole package. Repeated selection of the same checkpoint means a
larger coefficient, not duplicate inference or another independent seed.

For planning, retain runtime headroom while EPYC measurements are absent; this
does not change the user's 60-minute ceiling. Compare paired exact packages on
one local core and calibrate against actual portal runtime receipts. Intel/EPYC
differences and shared-core variation remain uncertainty, not guaranteed ratios.
Keep model membership fixed: no wall-clock governor that changes predictions.

## Analytical starting point and its exact limits

For one target, fixed official weights w=abs(clip(y,-2,2)), and inactive prediction
clipping, form weighted centered prediction covariance C, prediction/label
covariance c, and clipped-label variance v_y. For coefficients a:

    r(a) = (c^T a) / sqrt(v_y * a^T C a)

When C is positive definite and the covariance is achievable, the unrestricted
optimal direction is solve(C,c). Do not explicitly invert C. With singular C,
remove zero-variance directions or use a declared regularized initializer
solve(C+lambda*I,c); the latter changes the problem and is not an exact optimum.

The single-target nonnegative version can be represented as an SOCP:

    maximize c^T a
    subject to ||L a||_2 <= 1, a >= 0, L^T L = C

Normalize the nonzero result afterwards using positive-scale invariance. This
requires the same inactive-clipping assumptions. The second-order-cone framework
is described by [Lobo et al.](https://stanford.edu/~boyd/papers/socp.html);
the weighted-Pearson formulation here is our derivation, not a result claimed
for the competition by that paper.

For s=a^T C a>0, the unclipped derivative is:

    grad r = c/sqrt(v_y*s) - (c^T a)*C*a/(sqrt(v_y)*s^(3/2))

The official objective clips the mixed prediction and averages two correlations
with generally different covariance matrices. Shared weights therefore do not
have a general C^-1*c closed-form solution. With x=clip(P*a,-2,2), use
dx/da_j=P_j*1(|P*a|<2), weighted covariance/variance derivatives, and sum the
two target gradients. Exact boundary kinks and zero-variance cases need explicit
handling. A local numerical optimum is not a global guarantee.

Use [SciPy SLSQP](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
with the real-arithmetic clipped objective and analytic Jacobian, a>=0 and sum(a)=1.
Compare only three deterministic starts: equal weights, current fixed weights,
and the analytical initializer. Cap at50 iterations. Keep equal weights as the
control; test a fixed shrinkage toward equal weights to reduce fitting variance.
An optional per-target coefficient vector is a separate higher-complexity arm.
The analytic derivative applies to this smooth relaxation, not the float32
rounding step in the computational scorer. The unchanged official scorer casts
predictions to float32 before clipping. Independently score and rank final
candidate weights through that exact path, and ultimately the packaged native
mixture, including near-constant/ULP-sensitive cases. A close synthetic numerical
match does not prove computational identity. Unrestricted negative weights and causal gates are deferred until simpler
positive stacking shows conditional value on another evaluation partition.

## Discrete selection before expensive weight fitting

Use bounded forward selection maximizing marginal official-score gain of the
whole ensemble, with measured incremental runtime and package costs recorded.
Consider a weaker standalone member when it has conditional value; do not add
it merely for architecture novelty or low raw correlation. At each step include
the existing ensemble as an option, so adding a model is never compulsory.

[Caruana et al.](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)
provide metric-based forward ensemble selection and discuss selection overfit.
For our correlated sequences, use complete components rather than independent
rows. Include model/precision selection and weight fitting inside the calibration
side of component cross-fitting. Report performance on its held-out counterpart
and other compatible folds. These historical folds remain exposed development
data; cross-fitting does not restore a pristine holdout.

Initial comparison budget: at most12 frozen endpoint/policy candidates, three
solver starts per selected pool, <=4 distinct recurrent bodies where measured
cost permits. Reopen larger pools on a concrete compression/runtime mechanism.
This is a bounded endpoint comparison, not an open-ended Optuna/random sweep.

## Precision is a model option, not a speed assumption

Maintain three separate quantities: storage rounding, inference arithmetic, and
the resulting complete-mixture score. Current Q8 files may store FP32 arrays;
selective FP16 storage is not FP16 inference. Q4-rounded FP32 arrays executed by
Q8 kernels are not native INT4 and cannot establish a low-bit speedup.

Candidate mechanisms:

1. Q8 for all bodies, FP32 biases/final readout/normalizers: current control.
2. Selective FP16 disk transport with the same Q8 runtime: only if ZIP size is
   binding; exact stored-value changes still require grading.
3. Mixed precision by matrix: protect sensitive recurrent/projection operations,
   first screen lower precision in less-sensitive dense/readout groups, then
   verify complete20k-row recurrent trajectories. Do not assume which group is
   insensitive before measuring it.
4. Calibration-aware lower-bit rounding and compensation rather than max-range
   rounding alone. [GPTQ](https://arxiv.org/abs/2210.17323) motivates second-order
   error compensation; [HAWQ](https://arxiv.org/abs/1905.03696) motivates precision
   allocation by sensitivity. Their Transformer/CNN results do not establish
   performance for our GRU or EPYC kernels. Calibrate on permitted fitting data;
   keep CV evaluation groups outside calibration.
5. Short GPU QAT/distillation recovery only after a low-bit mechanism offers
   actual resource headroom and loses recoverable quality. QAT is a new fit with
   explicit ancestry, schedule and clean evaluation; it is not currently admitted.

[HAQ](https://arxiv.org/abs/1811.08886) supports using measured hardware feedback
instead of bit count/FLOPs as a latency proxy. Q5–Q7 may be useful storage formats
while running Q8, but custom packing/unpacking may cost more on EPYC; benchmark
before claiming a gain. Never import an LLM kernel's speed claim into GRU serving.

## First bounded interaction comparisons

| Arm | Question | Fixed control / requirement |
|---|---|---|
| J1 | Can exact-metric positive weights beat equal weights? | Same Q8 members; component-cross-fitted selection; analytical/SLSQP initializer and fixed shrinkage |
| J2 | Do extra seeds improve quality per measured CPU second? | Strong matched recipe,1/2/3 seeds; distinct seed means and marginal complete-mixture gain |
| J3 | Does another family beat the next seed? | Same body/runtime budget, comparable source quality; no selection on raw correlation alone |
| J4 | Can sensitivity-based mixed precision preserve quality? | Same members; current Q8 versus calibrated matrix-specific rounding; whole-sequence fidelity and held-out score |
| J5 | Does a larger compressed blend beat a smaller Q8 blend? | Both endpoints feasible; compare on exactly the same CV groups after actual serving precision |
| J6 | Can several readouts share a body at useful quality? | Exact own trunk equality and fixed head weights; current HFT changes sign across F0/F1, so no automatic inclusion |

J1 numerical implementation can proceed now. J2/J3 full cross-fold comparison
waits for matched clean F0 students already in delivery. J4/J5 start with bounded
fidelity/cost screens and need independently reviewed frozen recipes before GPU
QAT or native changes. These are proposed comparisons, not launched fits or gains.

Use existing metric, prediction-bank, provenance, scoring and component tools;
stream/memmap arrays with explicit support limits. Precomputed moments accelerate
the unclipped initializer, but do not replace the nonlinear clipped evaluator.
Do not materialize every full prediction bank in RAM. Local support stays within
5CPU/18GBworking/20GBhard/no swap; one GPU and5GB observed VRAM. Refresh Kaggle
quota before dispatch. No new paid spending path is opened by this plan.

## Evidence that prevents repeating old uninformative work

Three teacher seeds gain about.0019 over the mean individual on each exposed
F0/F1 panel; adding a different family to a stronger endpoint has often been
small/null or negative. Earlier fitted stacking/covariance/calibration barely
improved or underperformed equal weights. These conclusions concern those pools,
not all future combinations. The new test changes the served precision/cost
interaction and uses the exact official objective, not a public regression.

Q8 loss on the ten-sequence H390 fidelity panel is.0000709 versus FP32; the tested
Q4 rounding loses another.00180125. Those panel differences cannot be subtracted
from ensemble gains measured on another fold. New four-body mixture promotion
needs direct same-group evidence and a feasible exact ZIP, not added proxy deltas.
The public metadata ridge remains worse than the recent-five chronological
baseline; no new public numerical forecast or public-like fold is established.

Owner: root for mathematical plan/allocation; Luna/max numerical actor for J1;
separate independent review before real-data optimization/promotion. Existing
cache/profile/student owner and Kaggle successor owner remain active. Results,
retractions and exposure will be appended to ledgers before any candidate release.

## Actual competition runtime update

User reports C (`h390_s3_s4_q8_z4half_20261012.zip`, portal1ya21vpd) scored0.6671 in36m36s,642MB. This is user-transcribed rather than browser-verified. It leaves23m24s under60min for this exact package. Both less costly frozen options remain unsubmitted. Prioritize useful conditional additions to C and weight optimization before accepting Q4 loss; exact next-package runtime is still unmeasured. One C result cannot disentangle seed/family contribution or establish a public-score predictor. See monitorR20/USER_PUBLIC_C_RESULT_R1.json.
