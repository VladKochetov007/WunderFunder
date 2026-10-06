# Independent specification critic review

Actual OpenCode model: `opencode/nemotron-3.5-lightning-free`. Both successful calls exited0. First attempt failed at CLI argument parsing before a model result; original error is preserved. No actor proposal, implementation or results were provided to this critic. Raw visible outputs remain unchanged; private reasoning is not archived.

The first response mostly restates the prompt. The follow-up has concrete but incorrect expected outcomes and is **not implementation approval**.

- Two-row permutation preserves the only Euclidean pair distance. With one sampled pair, normalization makes every nonzero distance nearly one, so this geometry carries almost no relative-distance information. Use multiple pairs and at least three noncollinear rows for a permutation diagnostic.
- Normalization with additive epsilon is only approximately scale-invariant; translation invariance does not imply strictly positive loss. Probe small and large scales in float32 and mixed precision.
- A collapsed student compared with a nondegenerate teacher gives a Smooth-L1 loss near0.5 under one normalized pair, not necessarily greater than1. Degenerate teacher relations require an explicit policy; zero relational loss alone proves no task utility.
- One active masked row yields no eligible pair and should give a finite graph-connected zero. It cannot support a relational inference claim.
- Teacher gradients must be checked after backward; `.grad is None` before backward tests nothing.
- Repeated equal scalars do not establish unchanged global random state. Snapshot Python, NumPy and Torch states before and after, including the disabled path.
- Nonfinite inputs must be rejected explicitly. Returning NaN is not acceptable.

Keep original-label and point-teacher losses unchanged. Check the zero-coefficient path as object-identical and without extra random draws. Statistical intervals spanning zero remain insufficient evidence rather than universal rejection.
