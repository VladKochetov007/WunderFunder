# Research choices for the remaining month

This is a prospective controller strategy, not a claim that a new model wins.
Today's user-authorized execution ends23:00 local. Future-month work below is an
agenda, not authorization for jobs after that endpoint.

## Selected now

Three executable actors and a blind critic are running (round2/agents.json):
loss estimator, target stopping laws, and strong-GRU information use. They attack
different assumptions and have disjoint write scopes. Existing width320 and
additional-fold combo jobs remain useful controls, not substitutes for discovery.
After independent reports freeze, select by discriminating evidence and reusable
artifacts, then ask the critic/reproducer for a real test before a full fit.

## Mechanism-changing options after evidence

| Option | Representation change | Cheapest decisive test | What would stop it |
|---|---|---|---|
| Path-functional teacher | Teach a causal student a distribution of future event-path functionals, rather than only two realized scalar returns. | Identify a bounded stopping family on supplied TRAIN paths with known/unknown/censored coverage and synthetic ground truth. | Shifted-path control ties, or reconstruction depends on future-picked class/coverage. |
| Event simulator as a teacher | Learn transition distributions on anonymized observable states, simulate a small event tree, and distill its conditional path expectation. | Synthetic planted-process recovery plus one fixed conditional next-event diagnostic with a sham generator. | Good marginal next-code fit but bad conditional path calibration; no causal prefix sufficient state. |
| Shared normalization objective | Replace batch-dependent ratio weighting with a declared target-wise estimator while preserving updates, row membership and EMA cadence. | Actual chunk normalizer measurement plus clustered-regression positive/negative controls. | Normalizer variation is negligible, or measured estimator difference is benign at equal capacity. |
| Information-preserving quotient | Identify which encoded levels/history are redundant and learn the remaining transition state, allowing a smaller trunk on one CPU. | Strong-model identity-checked history interventions and a synthetic planted-history instrument. | Intervention response is entirely out-of-distribution or the state reduction discards predictive information. |
| Prefix uncertainty curriculum | Change the training problem from repeated identical views to a sequence of causal information states, with hindsight used only for supervision. | Synthetic memorization task plus fixed prefix/view control with equal usable rows and identical inference. | Gain comes from changed row membership/update count or causal suffix leakage. |

These are not extra hyperparameter slots. Some touch rejected families; they
require the stated new observable to justify a genuinely different mechanism.
Do not relaunch class calibration, carry, phase or universal event clocks under
new names. An unsuccessful particular configuration does not prove a theorem
against every representation in its family.

## Experimental design that compounds knowledge

Adaptive reuse of validation can overfit the evaluation itself; ordinary
fixed-procedure guarantees do not automatically survive interactive selection.
The reusable-holdout papers propose specific mechanisms for this problem.
We have not implemented those mechanisms and cannot retroactively assign their
guarantees to already exposed folds. Primary sources read:
[Dwork et al., NIPS2015](https://papers.neurips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html),
[Dwork et al., Science2015](https://research.ibm.com/publications/the-reusable-holdout-preserving-validity-in-adaptive-data-analysis).

Our practical choice is to freeze mechanisms/controls before new scores,
distinguish discovery from transport, retain all failed trials, and reserve
independent implementations or paired seeds. This is a policy motivated by that
literature and our E-343 evidence, not a certified multiple-testing correction.
No dataset is called fresh without an exposure audit. Additional folds2/3 were
previously explored and are an additional-condition test, not a private holdout.

For each promoted family maintain a reward vector: validity, observed metric
transport, mechanism identification, novelty, runtime/memory and reproduction.
A modest or null score can still justify a reusable generator or instrument;
it cannot justify a leaderboard-improvement claim. A small proxy score gain with
poor transport can be a lower-value month-long direction than a falsified model
of the mechanics that sharply reduces the search space.

## Today: admission decision

Current width320 run began16:21. Pending width320 fold1 and combo folds2/3 are
sequential, with automatic scoring after each pair. Measured width epochs take
~16.2min. Original matched combo runs were faster. Any new full run needs a
reviewed command, measured cost and completion/export allowance before23:00.
Do not start an optimizer bundle before the deferred GPU/fused/EMA parity gate.
If a new representation deserves tomorrow-scale implementation, preserve a
prototype and experiment card rather than forcing an invalid rushed fit tonight.
