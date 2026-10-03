# Preregistration: finite-graph random credit with inverse survival

Written before executable experiment results. Tiny CPU instrument only; no fit,
GPU, trainer edit, actual capture/data, launch/receipt or new scientific arm.
Primary source read: Tallec/Ollivier arXiv1705.08209 abstract only. Its randomized
truncation/compensation motivation is used; no full-paper formula or implementation
identity is claimed. Our independent derivation concerns a finite fixed graph.

Graph: h[t]=F(theta,h[t-1],x[t]); same FIXED theta at every t, initial state/input
fixed, local losses differentiable. J[t]=d h[t]/d h[t-1], K[t]=partial h[t]/partial
shared theta holding incoming hidden fixed, c[t]=partial local loss/partial h[t].
Full reverse adjoint: a[t]=c[t]+J[t+1]^T a[t+1]. On each recurrent backward edge
keep independent Bernoulli Z[e] with prespecified p[e]>0, unrelated to graph/loss;
replace J[e]^T by Z[e]/p[e] J[e]^T. Forward states/loss never change. Sum K[t]^T a[t].
For any finite credit path, its multiplier has expectation1; linearity restores
full BPTT. No weights change inside graph; no adaptive keep or correlated decisions.

Fixtures fixed in code before run: scalar terminal h=w*hprev+b at(w,b,h0)=(1,0,1)
with2 steps and p=.5; long same recurrence11steps/10edges with p=.25/.5/.75;
vector2D linear recurrence4steps with shared A/b parameters, mixed local linear
losses and nonuniform p=(.5,.75,.25); scalar polynomial4steps h=a*hprev^2+b+x
with mixed local losses; first-step-only loss null with3steps. All numbers rational.
Each distribution enumerates ALL masks, at most2^10=1024, not Monte Carlo draws.
Oracle: independent FORWARD shared-parameter sensitivities over exact Fractions.
Compensated and ordinary detached gradients: reverse local Jacobians. Require exact
Fraction expectation equality to oracle for compensation; ordinary bias nonzero in
terminal/mixed planted controls, exact equality in first-loss null. p=1 exact null.
Independent closed-form long-chain coordinate variance=sum_{i,j=0..10}
(p^(-min(i,j))-1); require exact enumerated equality. Report cost proxies, variance,
max gradient amplification, probabilities and support, never claim measured speedup.

Precision: exact rational expected-gradient equality is the primary oracle test.
Additional float64 fsum expected gradient error <=256*epsilon*max(1,L1(analytic
full gradient)); fixed machine-rounding rule, not estimated/fitted from discrepancies.
Counterexample detection: exact scalar rational inequalities; vector nonlinear
update gap must exceed the same analytic-scale machine tolerance. No training
benefit/noise bound is fitted. All failures retained, no threshold or fixture retry.

Update witnesses: scalar2-step terminal h has compensated gradients1/3 with equal
probability and full gradient2. Test clipping cap2 and Adam-like first-step
map g/(abs(g)+1), zero incoming moments. Vector2-step case(w,b,h0)=(1,1,0)
gradients(1,1)/(1,3), full(1,2): global-L2 clipping cap2 and coordinate Adam-like
map. Transform AFTER unbiased gradient; compare E[T(g)] with T(E[g]). Also linear
SGD null. Adam-like map is not a claim about actual trainer's warm Adam/Muon/scaler.

Planted assumption failure:3-step scalar, edges perfectly correlated with equal
all-cut/all-keep probability. Compensation still uses marginal p=.5. Its expected
gradient must NOT equal oracle; inverse marginal survival is insufficient because
joint path survival differs. Report this as expected detected failure, not a failed
code check. No global RNG draw is needed anywhere.

Resource/time: stdlib Fraction/math only, tiny dimension<=2 and parameter count<=6,
no wide tensors or data loader. Every execution/lint uses run_cpu.sh round4-capture
1core/2GiB/900s. Expected seconds, not fitting. Freeze code/results/report/STATE/
hashes and bindings; earlier scopes and frozen month/index artifacts untouched.
