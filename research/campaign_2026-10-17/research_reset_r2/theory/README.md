# Mathematical and literature distinctions

- [Persistence and prediction](PERSISTENCE_AND_PREDICTION.md) derives the exact
  clipped-label-run weighted-variance identity and explains why a constant
  realized target need not justify a constant causal estimate. It also records
  the pre-score interpretation table and the existing WMSE consistency term.
- [State clocks and predictive queries](STATE_CLOCKS_AND_PREDICTIVE_QUERIES.md)
  distinguishes observation, state-write, output, label and gradient clocks;
  checks five primary sources; and refines existing KD/event-auxiliary proposals
  without claiming a new fitted result. Abstract/formulation inspection is
  distinguished from full-paper reproduction.
- [Sequence decomposition](SEQUENCE_DECOMPOSITION.json) reconstructs all84
  target-specific policy scores from saved raw moments, matching the canonical
  centered grader within5.56e-16. Its within/between covariance terms share the
  global denominator; they are not subgroup correlations or causal effects.
  [Derivation script](derive_sequence_decomposition.py) binds the source grade
  and sufficient-statistic hashes; [numerical test](test_sequence_decomposition.py)
  checks100 masked synthetic fixtures against the public canonical scorer,
  with no model or optimizer calls.
- [Plot renderer](plot_output_audit.py) and [figure index](../plots/README.md)
  preserve the observed deltas and conditional uncertainty without fitting or
  choosing a filter.

Both notes belong to [R2](../README.md). Predictions and labels are not available
to an inference policy beyond the rules stated there; oracle diagnostics never
become deployable features.
