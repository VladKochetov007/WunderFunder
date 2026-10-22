# Fixed numerical bases for a causal Transformer

Compare raw inputs, uniform-bin and quantile-bin scalar encodings while keeping
one temporal token per row. This is a concrete child of [PR #8](https://github.com/VladKochetov007/WunderFunder/pull/8)
and the numerical-basis ladder in [PR #31](https://github.com/VladKochetov007/WunderFunder/pull/31),
not a newly discovered model family. The two binned arms differ only in knot
placement. Raw inputs remain available in every arm.

**Proposal only: no encoder implementation, device profile, training or score.**
The receiving engineer must integrate the encoder with the existing
[PR #32 trainer](https://github.com/VladKochetov007/WunderFunder/pull/32), bind
source-clean data/splits and pass receiver-side acceptance. This packet does not
provide a fake training command or a plan printer presented as training.

- [PR.md](PR.md): mechanism, six-fit comparison, decision and handoff.
- [recipe.json](recipe.json): explicit arms, fit counts, source pins, leakage and
  implementation blockers. It is a research recipe, not a trainer config schema.
- [Public search and submission triage](../../unified/sources/CONNECTOME_PUBLIC_SEARCH_20261022.md):
  exact-round versus predecessor models, actual team results and candidate order.

Parent scientific ideas remain open. Existing broad proposals and their historical
CP30 plans are preserved; this child uses the current six-total-fit screen policy.
