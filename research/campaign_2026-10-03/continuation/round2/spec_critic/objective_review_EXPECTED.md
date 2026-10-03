Independent expressions fixed before opening objective_math.py, estimator_control.py,
analyze_probe.py or sham_probe.py. Based on actual trainer/source clauses, not imports
from an actor helper. Report/DERIVATION/PREREG have been read as claims to audit.

For a scheduled chunk, let v_i be the final row weight, Y_ij=clip(raw_y_ij,-2,2),
w_ij=v_i*abs(Y_ij), W=sum_i,j w_ij, R=sum_i v_i, and N=sum_i v_i*K_i.
All denominators below use max(mass,1e-8), and targets/weights are detached.
T is the clipped arithmetic average of scaled cached teachers, NOT the average
of individually clipped teachers. For bootstrap h8, A_i requires next RAW PAIR
change strictly greater than t+8 AND t+8<T_sequence. L_i=A_i*T_(t+8)+(1-A_i)*Y_i.
B_i=(1-alpha)*L_i+alpha*T_i; cached-teacher main is H/W with H=sum w*(f-B)^2.
Its output derivative is 2*w_ij/W*(f_ij-B_ij). Absolute raw Y or absolute B
must not replace abs(clipped original Y) in the coefficient.

With ema_metric_weight=false, EMA loss is lambda_e*sum_i,j v_i*(f_ij-U_ij)^2/R.
cw has shape (batch,time,1), so its sum is R despite broadcast to TWO outputs.
Derivative is 2*lambda_e*v_i/R*(f_ij-U_ij), with U UNCLIPPED.
Relative EMA/main quadratic coefficient per output is lambda_e*W/(R*abs(Y_ij))
on positive target magnitude. No universal factor1/2 or denominator2R exists.
With ema_metric_weight=true, numerator coefficient gains abs(Y) and explicit
consistency_scale2; denominator becomes W and derivative is 4*lambda_e*w/W*(f-U).

Structural loss is lambda_s/R*sum_i v_i*(CE_spread+CE_branch+CE_class), no /3.
Each categorical logit derivative is lambda_s*v/R*(softmax-onehot).
Next-change loss is lambda_n/N*sum_i v_i*K_i*((u-(log1p(nextDrows)-3)/2)^2+BCE(a,up)).
Its MSE derivative has factor2, BCE derivative does not. K concerns a future
RAW TARGET DIFFERENCE change, not merely either target pair changing.
These heads share the trunk; they cannot generally be replaced with a scalar
change to the primary-output optimum. U evolves with lagged EMA updates.

The actual cached-teacher if/elif comes before the running-normalizer branch;
once teach is not None, loss_norm running cannot affect H/W or update norm_ema.
This is a documented branch limitation, not evidence of numerical corruption.

Let Hgradient_C be the unnormalized fixed-student main parameter gradient.
gSN_C=Hgradient_C/W_C. D=mean_C W_C. Then s_C=W_C/D gives s_C*gSN_C=Hgradient_C/D
EXACTLY. Permuting the finite scalar multiset matches multiplier counts but not
gradient norms, mass alignment, multi-loss directions or optimizer steps.
For uniform permutation pi over n slots, E_pi mean(s_pi*gSN)=mean(s)*mean(gSN)
=mean(gSN), since D=mean(W). Difference aligned minus that expectation is the
finite covariance mean((s-1)*gSN). This says nothing about an optimum unless
the gradient functions and model/conditioning are identified beyond one proxy.

TRAIN-only v uses valid*(propensity^2+.03)*since_weight. Validation-in-fit first
replaces propensity weights on masked sequences by c*valid*(scored+.03), with
one batch-dependent scalar matching masked aggregate mass; since multiplies later.
Hence omitted validation rows can change W/R and relative losses after since.
Temporal chunks and connected sequences share information; leave-one-batch-out
prediction scores are not component-independent validation.
