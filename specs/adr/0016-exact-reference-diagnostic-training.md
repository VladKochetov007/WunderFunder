# ADR0016: exact teammate-source diagnostic training

2026-10-13 successor: [ADR0035](0035-official-team-code-and-weight-sharing.md). Diagnostic-only and fresh-weight restrictions superseded for future official-team work; the historical experiment remains unchanged.

Status: accepted, 2026-10-08 Kyiv. Narrow diagnostic exception to ADR0012, based on the latest explicit user instruction.

## Basis

After reporting H230 public0.6649, the user requested: “try to train exactly maxim's code from his branch and make eda of our and his model difference”. The purpose is to distinguish implementation drift, training membership and model/data behaviour from an incomplete reproduction.

## Decision

An isolated diagnostic may execute the identified teammate training source unchanged, with our freshly trained weights and declared permitted input data. Freeze its commit, exact commands, dependencies, fitting membership, all teacher/initializer ancestry, checkpoint rule and numerical policy. Identify the actual source of the successful0.6674 recipe rather than assume the latest branch headline or Transformer commit produced it. A source inspection or model profile is not a replication result.

Do not initialize from or distil teammate-trained weights. Train any required teacher/base models independently. Preserve source-local diagnostics separately from our implementations and record adaptations needed solely for execution; any semantic adaptation becomes an identified variant rather than an exact-source result.

This exception authorizes reference training and analysis, not copying teammate code into a submission or performing a cosmetic rewrite. Production training/serving/build code remains independently implemented under ADR0012. Keep weights, artifacts and packages labelled by reference versus independent implementation ancestry.

Run independent recipe, data/metric and prediction-behaviour investigations. Compare matched fitting sets and controls before attributing the public gap to a bug. H230 is a fold-trained single; the teammate public endpoint is reported as an all-data pair. This distinction is a concrete confound to test, not proof of the explanation. Historical incumbent membership corrections remain valid despite the new public result.

## Delivery

Require a runnable frozen reference diagnostic, representative device profile, useful bounded fit, predictions and side-by-side official/behavioural evaluation. Use GPU/TPU for model training and resource-bounded CPU/marimo for analysis. Save EDA images under the investigation lineage's plots directory. Record all failures and partial results; no public replication or new best until actual evidence supports it.
