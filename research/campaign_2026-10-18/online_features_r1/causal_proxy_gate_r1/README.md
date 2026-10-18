# Causal pseudo-label expert gate R1

## Question and status

Does a feature-only estimate of the previous-row label help a causal, within-
sequence mixture of the frozen TL3 and R3 predictors? The proxy was trained
without evaluation labels and scored 0.521679 WP on its separate, exposed-F0
feasibility panel. Here the pseudo-label produced from `x[t]` updates relative
expert-error state from the experts' predictions at `t−1`; the new mixture is
used for row `t`. State resets at each sequence. The target horizon and label
maturity are undocumented, so this tests pseudo-label utility, not actual
realized feedback.

This is an exploratory local screen, not clean validation: its 32 source-
qualified F0 components are component-disjoint from proxy fitting but have
been repeatedly exposed. TL3 ancestry remains unknown and potentially fitted.
The existing fixed 75/25 blend underperformed TL3 on all exposed F0, so TL3 is
the primary comparator and the fixed blend is the matched prior control. This
screen is deliberately allowed to be inconclusive; it does not close the wider
online-adaptation direction.

## Frozen treatment

The predictor script receives only the frozen proxy predictions, the two
source-qualified expert prediction caches, sequence boundaries and row IDs.
It does not read target labels or the official mask. It writes six endpoints:
TL3, R3, fixed 75/25 TL3/R3, and proxy-driven gates with 256, 2,048 and 8,192
step half-lives. Each gate starts from 25% R3, waits 128 proxy observations,
then uses an exponentially weighted, pseudo-target-magnitude-weighted squared
error ratio per target. The R3 weight is updated by a fixed softmax-style
relative-loss rule (`eta=2`) and clipped to [0.05, 0.50]. No evaluation target
is used to select an update, gate, or weight.

At time `t`, the proxy for `y[t−1]` is compared to each expert's clipped
prediction at `t−1`; this updates weights before combining predictions for
`t`. The first scored position (row 99) uses the fixed 75/25 prior because no
following state is available yet. All gates reset at every sequence boundary.
The offline grader alone reads validation targets/masks and uses unchanged
`global_wp_clip2_v1` over rows >=99. Exact source/cache identities and all
outputs are bound in the recipe and execution receipts.

## Evidence decision

The frozen screen completed. All three adaptive gates were effectively equal
to the fixed 75/25 control and remained below TL3 on the same 32 exposed F0
components. The best gate's paired improvement over the fixed control was
only +0.00004795 WP, with conditional component interval
[−0.00003098,+0.00012853]. The fixed blend itself was −0.00237815 below TL3.
See [results](RESULTS_R1.md), the [recipe/code audit](RECIPE_AUDIT_R1.md), and
the [execution receipt](execution_receipt_r1.json) for complete evidence.

Do not make a separate B300 draft for this specific proxy-gate mechanism. The
screen narrows this single linear-proxy / cached-expert treatment; it does not
reject online learning, causal state features, student-side adaptation, or
genuine matured feedback if allowed by the evaluation interface. The existing
causal-state teacher/student draft PR #17 is a separate proposal and has no
validation from this screen.

## Files and artifacts

- [Frozen recipe](recipe.json) pins the proxy, both expert caches, source
  membership, gate constants, endpoint IDs, and evidence limits.
- [Recipe builder](prepare_recipe.py) validates cache hashes, sidecars and
  split membership without reading labels.
- [Causal prediction builder](build_predictions.py) applies gates using only
  current/past features represented by the frozen proxy and prior expert
  predictions.
- [Independent grader](grade_gate.py) reconstructs each gate independently,
  checks prediction parity, then scores fixed predictions with the official
  metric and conditional component bootstrap.
- [Recipe/code audit](RECIPE_AUDIT_R1.md) compares the frozen equation, row
  alignment, cache identities and label-blind code path.
- [Execution receipt](execution_receipt_r1.json) records hashes, complete
  commands, resource peaks and zero OOM deltas.
- [Results](RESULTS_R1.md) records all six endpoint scores, paired component
  intervals and the scope-specific decision.
- [Independent evidence review](../../../unified/reviews/CAUSAL_PROXY_GATE_EVIDENCE_REVIEW_20261018_R1.md)
  checks endpoint support, sequence/mask ordering, resource receipts and claim
  scope without rerunning the experiment.

Model outputs and prediction payloads stay in physical scratch at
`/home/vlad/development/wunder/research_tmp`; only small manifests, code,
diagnostics and receipts belong in this research folder.
