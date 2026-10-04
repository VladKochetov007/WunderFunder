# H090 factorized branch/U readout

Historical reconciliation: E071/E072 trained504-code CE heads with a rational
|y|-moment readout. Plain CE scored0.6489, tilted CE0.6536, tilted CE+wMSE0.6645,
versus wMSE0.6651, on the older H128 one-epoch setup. Claims that H046 had never
received a fit are too broad. This protocol retests those families on the modern
frozen trunk while adding a distinct conditional factorization. Prior negatives
are controls, not exclusions.

The candidate distribution is
P(class|h) × P(spread-code|h) × P(branch|h) × P(U-bin|branch,h).
There are4 class,7 spread-code,3 branch and5/8/5 branch-conditional U-bin logits,
32 total, inducing a normalized distribution over the existing504-code alphabet.
Conditional on h, class, spread-code and branch/U are assumed independent, while
U-bin depends on branch. This is a statistical factorization of supervision codes;
the archived target inversion is not asserted to recover true physical spread.
No target class, spread or branch is supplied at prediction: all probabilities
come from the causal frozen state. Evaluation classes are read only for descriptive
score strata after predictions have been formed.

The prespecified arms are factorized plain CE, factorized tilted CE+wMSE, old
joint504 plain CE, old joint504 tilted CE+wMSE, and direct wMSE. The near-tie older
hybrid is included rather than using only the weakest CE comparator. All heads
use identical E415 A/B rows and fitting weights, state normalization fitted on
the fitting partition, AdamW lr0.001/weight-decay0.01, the same256 minibatch updates
of at most512 rows, seed271828, and the final update with no validation selection.
One GELU hidden layer is resized to fit65,536 learned parameters within2%; exact
counts and a common minibatch-index hash are reported. The frozen trunk is shared.
E415 fitting weights are unit weights: this is not an exact replication of the
historical end-to-end propensity² training protocol.

All categorical arms share the same fitting-only empirical moment tables:
M1=E[|clip(y)|clip(y)] and M0=E[|clip(y)|]. A code gets8 fixed pseudo-observations
from its fitting class/branch mean; if unavailable, use its fitting branch mean,
then the fitting global mean. Thus unseen codes have defined support behavior,
with no evaluation-table fitting, oracle formula, or output-dependent rejection.
Weighted code support, unseen code count and low-support count are exported.
The tilted arms divide code probabilities by mean(M0) before the rational readout,
matching the older weighting interpretation. This shared sparse-table smoothing
is an explicitly new screen policy, not a claim to reproduce historical scores.

Every report includes the untouched predictor, simple global affine calibration,
both factorized candidates against all controls, pooled WP and class-specific WP,
paired component bootstrap intervals, learned weights, moment tables and all
predictions. Same-state A/B partitions are exchanged for a reciprocal screen.
No head is selected on either reported evaluation partition. An apparent gain
over an impaired refit does not justify promotion: improvement over untouched
and affine predictions, mechanism controls, class3 behavior and independent
confirmation must be assessed together. End-to-end factorized training and native
inference remain subsequent implementation work; this milestone is the frozen
screen only.

`scripts/feature_branch.py` is executable on CPU or CUDA, with draft specs under
`branch_drafts/`. Every draft must be sealed and admitted before scientific fitting.
Bounded synthetic tests check normalized/differentiable probabilities, sparse
support, current-trunk parameter budgets, equal fit steps, and prediction invariance
to changes in evaluation class/target labels.
