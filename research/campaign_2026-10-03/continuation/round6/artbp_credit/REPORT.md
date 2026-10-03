# Exhaustive finite-graph compensated credit oracle

This new tiny measurement instrument passes the preregistered fixed-graph oracle,
ordinary-truncation bias/null controls, variance identity and scalar/vector nonlinear
update counterexamples. No fit/GPU/actual capture/trainer change or scientific
training approval. It is distinct from R5's common-macrostep deterministic detach
control: this instrument exhaustively integrates random credit masks and shows the
failure of expected-update equivalence after nonlinear transforms.

## Source and claim boundary

Read [Unbiasing Truncated Backpropagation Through Time](https://arxiv.org/abs/1705.08209),
Corentin Tallec/Yann Ollivier, arXiv1705.08209, ABSTRACT ONLY, plus the local
MEMORY_LITERATURE_ADDENDUM.md. The abstract motivates variable truncation and
compensated backpropagation for unbiased gradients and reports synthetic/character
model experiments. Neither PDF/full-paper formulas nor their algorithmic details
were read here. This is our independent finite-graph construction inspired by that
motivation, not an implementation/parity claim for the paper's ARTBP algorithm.

## Own chain-rule derivation

For fixed parameters theta shared over time, h[t]=F(theta,h[t-1],x[t]) and
L=sum local_loss[t](h[t]), define the local hidden Jacobian J[t], direct parameter
Jacobian K[t] with incoming state held fixed, and local loss derivative c[t].
The full adjoint is a[t]=c[t]+J[t+1]^T a[t+1]; the parameter gradient is
sum_t K[t]^T a[t]. Direct loss-to-parameter terms, if present, must be retained
separately unchanged; these fixtures have none. The initial state is fixed and
parameter-independent, and carries no missing initial-state parameter gradient.

Choose each backward edge's independent Z[e]~Bernoulli(p[e]), p[e] in(0,1]. Replace
that edge's propagation by Z[e]/p[e] J[e]^T. All forward states, parameters and loss
values stay unchanged. Expand a[t] as a finite sum of path products ending at local
loss derivatives. Each path uses every traversed edge once, and gains multiplier
product_e_in_path Z[e]/p[e]. Independence gives expected multiplier1, so linearity
recovers full BPTT. Alternatively downstream random adjoints are independent of the
new upstream mask, establishing the same finite reverse induction.

Sharing parameters does not violate this proof: K[t] is fixed by the unchanged
trajectory and its contributions add linearly. Nonlinear F also does not invalidate
it: local Jacobians are fixed before masks. Changing weights within the graph,
randomly changing forward hidden values, loss-dependent/adaptive mask probabilities,
correlated decisions without correct joint/path compensation, or missing paths would
require a different proof. Unbiased credit at old fixed weights is not automatically
a current-weight gradient of an online-updated trajectory.

## Implementation and independent checks

credit_oracle.py builds rational scalar/vector linear or polynomial trajectories;
forward_oracle propagates full shared-parameter sensitivities forward in time.
reverse_credit propagates compensated or ordinary adjoints backwards on exactly
that same graph. Immutable tuples/Fractions keep states/weights fixed. Every mask
and its exact product probability is enumerated, at most2^10=1024 per distribution;
there are ZERO random draws or Monte Carlo gradient estimates. Full gradient
expectation equality is exact rational arithmetic. Known two/eleven-step closed
forms and p=1 controls additionally check the analytic oracle. Float64 probability
integration uses fsum and a preregistered256*epsilon*max(1,oracleL1) bound; it is
not calibrated from a desired difference. Exact rational equalities are stronger.

The first-step-only null has no future credit and ordinary/compensated expectations
both agree exactly. Terminal and mixed local losses show ordinary detached bias
while compensated expectation agrees with full BPTT. One fixture has2D hidden state,
six shared matrix/bias parameters and nonuniform probabilities; one scalar transition
is polynomial. No network training or PyTorch/autograd import is needed.

## Gradient versus clipping/nonlinear optimizer expectation

The scalar two-step graph h[t]=w*h[t-1]+b at(w,b,h0)=(1,0,1), terminal L=h[2], has
full derivative2 in the w coordinate. With one p=.5 edge, its compensated w gradient
is1 or3 equiprobably: unbiased mean2. Treat b fixed for this scalar update witness.
Clipping at2 gives expected clipped gradient1.5 versus clipped full gradient2.
The Adam-like first-step map g/(abs(g)+1) gives expectation5/8 versus full2/3,
gap-1/24. This corresponds to zero incoming moments and an epsilon1 normalized
first-step transform; it is deliberately not the actual warm trainer Adam/Muon.
Actual parameter update would multiply by-lr, reversing the sign of the displayed
normalized-gradient gap. A frozen linear SGD map commutes exactly with expectation.

The two-parameter scalar-state graph(w,b,h0)=(1,1,0) gives compensated gradients
(1,1)/(1,3), full(1,2). These are noncollinear. Global-L2 clipping at2 gives expected
vector ((1+2/sqrt(10))/2,(1+6/sqrt(10))/2), versus
(2/sqrt(5),4/sqrt(5)) for the full gradient. The coordinate Adam-like map preserves
the first coordinate but has the same-1/24 second-coordinate expectation gap.
results.json retains actual float clipping gaps and exact rational Adam-like gaps.
Thus gradient unbiasedness does not imply expected clipping/optimizer equivalence,
unchanged trajectory, bounded variance, safe loss scaling, beneficial Muon behavior
or a competition score improvement. No actual optimizer is simulated here.

## Variance, cost and expected failure

The11-step scalar terminal graph atw=1 has full w and b gradients11. For constant p,
the compensated coordinate is sum_{k=0..10} S[k], S[k]=product of the final k
edge Z/p factors. Cov(S[i],S[j])=p^(-min(i,j))-1. The independent analytic variance
sum_{i,j}(p^(-min(i,j))-1) matches enumeration EXACTLY. Small survival probabilities
reduce average terminal backward depth but create rare heavily amplified credit;
variance can grow exponentially with path length. No universal variance guarantee.

All11 forward states are still computed, and this oracle stores the full graph.
Expected retained edges=sum p[e] is only a structural proxy, not useful backward
work in every loss layout. With terminal loss, expected active local VJPs are
sum_{k=0..10}p^k; expected active recurrent VJPs omit k=0. With losses at many times,
cuts need not eliminate local work. Enumeration costs O(2^E*T*(n^2+n*m)); n<=2,
m<=6,T<=11. Stored graph is O(T*(n^2+n*m)); masks process sequentially. The measured
execution is the exhaustive validation tool, NOT a benchmark of an optimized random
truncation trainer. No GPU memory/speed claim or full-data loading.

Planted failure: on the three-step terminal scalar graph, make the two masks
perfectly correlated (all-cut/all-keep equally likely) but still divide by their
marginal p=.5. Full gradient3 becomes expected compensated gradient4. The two-edge
path survives with probability.5, not.25, so inverse marginal products are wrong.
The instrument must detect this expected mismatch. It does; this is an assumption
failure witness, not a failed execution or evidence that independent compensation
is biased. Ordinary detachment also supplies deliberately biased controls.

## Preservation, reproduction and pending work

PREREG.md exists before computation; pre_experiment_bindings.json records its hash,
final implementation hashes and the unchanged local literature addendum before
experiment(). Failure tracebacks are preserved without relaxing thresholds. No
execution failure occurred in the completed run. HASHES.json binds final artifacts;
logs and exact rational results are retained. Reproduce run.sh in a NEW authorized
copy: runner refuses to overwrite existing results. Use the CPU wrapper1core/2GiB/
900s; actual body takes seconds with tiny stdlib objects, not fitting.

Original capture/index/month/actor/critic artifacts remain untouched. State drift
and actual AMP numerical audit are still NOT measured by this instrument. Next
legitimate step is independent review of the finite-graph assumptions/code/results,
not integration into the live trainer. Adaptive survival, finite-precision scaling,
variance control, realistic loss/parameter graphs and nonlinear optimizer behavior
would need separately preregistered controls before any training proposal. Physical
connectome identification, leaderboard gain and scientific promotion are absent.

Measured enumeration summary:

| Case | Masks | Ordinary mean | Full gradient | Variance trace |
|---|---:|---|---|---:|
| scalar_two_step | 2 | (Fraction(3, 2), Fraction(3, 2)) | (Fraction(2, 1), Fraction(2, 1)) | 2 |
| long_chain_p1/4 | 1024 | (Fraction(1398101, 1048576), Fraction(1398101, 1048576)) | (Fraction(11, 1), Fraction(11, 1)) | 4660080 |
| long_chain_p1/2 | 1024 | (Fraction(2047, 1024), Fraction(2047, 1024)) | (Fraction(11, 1), Fraction(11, 1)) | 11996 |
| long_chain_p3/4 | 1024 | (Fraction(4017157, 1048576), Fraction(4017157, 1048576)) | (Fraction(11, 1), Fraction(11, 1)) | 578.432691 |
| vector_shared_matrix_nonuniform | 8 | (Fraction(23453, 16384), Fraction(33565, 32768), Fraction(6165, 4096), Fraction(8811, 8192), Fraction(35945, 16384), Fraction(4371, 2048)) | (Fraction(603, 256), Fraction(1629, 1024), Fraction(403, 128), Fraction(265, 128), Fraction(1687, 512), Fraction(129, 32)) | 40.8971523 |
| polynomial_fixed_graph | 8 | (Fraction(466002612513, 4398046511104), Fraction(16031942707041, 8796093022208)) | (Fraction(1917726103487, 17592186044416), Fraction(1025839701163, 549755813888)) | 0.00407563187 |
| first_loss_only_null | 4 | (Fraction(1, 1), Fraction(1, 1)) | (Fraction(1, 1), Fraction(1, 1)) | 0 |

Body wall 0.565194s; peakRSS 20440KiB. Exact rational identities passed in every case; float diagnostics and fixed bounds are retained in results.json. No fixture/tolerance changes or failed execution.

Finalization note: the first lint command stopped before any experiment due to an unused Fraction import. lint_attempt1.log and run_probe_lint_attempt1.py.txt preserve it; only the unused import was removed. Prereg/fixtures/probabilities/thresholds were unchanged. Exactly one scientific enumeration ran. Final hashes include the fully completed run.log (after runner stdout), and prior landed index/report and month option output hashes are verified unchanged.
