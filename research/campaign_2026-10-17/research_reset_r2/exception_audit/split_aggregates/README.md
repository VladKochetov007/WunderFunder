# Recoverable split aggregates

- [Train checkpoint](train.json) and [valid checkpoint](valid.json) preserve each
  completed split of the successful full selected-column census. Their byte
  hashes are bound in the original [run manifest](../manifest.json).
- [Combined results](../results.json) add the full dataset summaries and source
  identities. Read the [audit report](../REPORT.md) for definitions and limits.

These checkpoints contain descriptive aggregates, not fitted models. This index
was added locally at integration; it does not change either checkpoint.
