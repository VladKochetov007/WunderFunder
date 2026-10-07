# Independent review of weak-complementarity proposal cards

Reviewer: `complementarity_card_critic` (gpt-6-luna, max reasoning), read-only.
Scope: source claims, callback legality, overlap with prior work, and cheapest
discriminating checks. This is proposal review, not experiment evidence.

## Source and serving corrections

Krogh and Vedelsby's ensemble-ambiguity identity is for convex regression under
squared error. It supports asking whether disagreement corresponds to useful
conditional signal; it does not predict improvement under the competition's
clipped weighted-Pearson metric, and it does not establish that the weakest or
least-correlated member should be included. [Primary paper](https://papers.nips.cc/paper/1001-neural-network-ensembles-cross-validation-and-active-learning)

The random library passage is a loose analogy, not evidence for model gating.
The Vertox article's summary of predictable momentum returns is its author's
wording. The cited Barroso–Santa-Clara paper studies time-varying momentum risk
and scales strategy exposure using realized volatility; it does not study
forecast-model reliability. [Library article](https://www.vertoxquant.com/p/momentum-has-its-moments-with-code),
[cited paper](https://doi.org/10.1016/j.jfineco.2014.11.010).

The official callback receives one current `DataPoint` per call, containing the
current row's 112 features. A model may keep earlier rows in internal causal
state and receives sequence resets, but it receives neither targets nor the
scoring mask. The inaccurate shorthand in `DIRECTED_SOURCE.md` has been
corrected in place. [Callback fields](../../../../../docs/submission_guide.md#L48),
[causal row rule](../../../../../docs/data_overview.md#L51),
[hidden targets and mask](../../../../../docs/data_overview.md#L78).

Row-level residual and error diagnostics are screening proxies only. Final
candidate promotion must compare the complete fixed endpoint using the official
`global_wp_clip2_v1` metric; its clipping, mask, and weights remain fixed.

## Card decisions

### Card 1 — revise: output-only residual satellite

This is materially different from E-513/E-514 direct portfolio averaging.
Narrow its first version to candidate and anchor outputs; hidden features add
serving/export complexity. Cross-fit anchor, candidate, and corrector by
connected group, including recursive checks of propensity inputs and
checkpoint selection. First test whether candidate-output features predict
anchor residuals on separate groups. If that screen passes, compare the full
corrected endpoint with an equally capable input-only corrector and a
shuffled-candidate control. Keep the result on exposed groups exploratory.

Tree placement: retain as a child hypothesis in the existing
`arm:complementarity-gated-blend`; do not duplicate the arm.

### Card 2 — pass as a distinct training-allocation hypothesis

This changes where student updates are spent and leaves the weak model out of
the served endpoint, so it is distinct from direct blending and inference-time
gating. The source's active-learning result selects examples to label; it does
not establish that extra updates on already-labeled disagreement examples help.

Cheapest check: from ancestry-clean group-OOF outputs, test whether disagreement
strata predict anchor error under the official scored-row mask and target
weights on disjoint development groups, compared with matched random strata.
That tests association, not learnability. Only after a positive diagnostic,
compare equal-compute disagreement sampling against uniform sampling and a
sham schedule. Freeze sampling probabilities and importance correction first;
report any weight clipping.

Tree placement: a separate proposed arm under `family:optimization`, with null
numerical prior and no launch authorization until a clean prediction cache and
compute slot exist.

### Card 3 — revise: input-state reliability gate

This overlaps the existing causal blend-gate arm and should remain under that
arm. Define the gate target from group-OOF predictions and training labels; at
inference the gate can use only legal current/earlier feature state. Compare a
learned state gate with a constant gate, a shuffled-state gate, and the
unchanged anchor, scoring the complete endpoint.

## Limits and allocation implication

All local validation folds have prior exposure. Group separation still helps
control leakage, but does not create untouched confirmation. The existing
E-514 pool and scalar rule remain closed. None of these cards is an experiment,
score observation, numerical prior, or public forecast. A candidate's low score
or low correlation alone is not evidence that it contributes; test its
cross-fitted target-aligned residual signal and its complete endpoint.
