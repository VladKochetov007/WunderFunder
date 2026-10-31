# ADR0044: mandatory chronological primary validation

Status: accepted, 2026-10-27. Authority: the user's explicit standing instruction
to always split train/validation exactly by the Predictorium temporal protocol.

## Decision

The primary protocol is `train | valid | public | small gap | private`.
Training precedes independent validation, which precedes the hidden test periods.
Use the provided train/valid file boundary for future primary comparisons.
Do not replace it with random rows, shuffled sequence folds or random component
folds. Whole sequences and any recovered connected components stay disjoint.
Keep validation out of fitting, normalizer/calibration fitting, teacher and
initializer ancestry, learned blend weights and any other learned preprocessing.

The primary evaluation covers the official validation membership and the
unchanged global scoring mask/metric. Report train and validation separately,
including each target. A small subset diagnostic must not be called the primary
full-validation endpoint. Repeated validation use remains selection exposure.

Sequence identifiers are shuffled, not timestamps. Do not order seq_ix or assume
Parquet row-group order is chronological. An additional internal temporal split
requires actual timestamp/order metadata and a frozen source-qualified manifest;
otherwise preserve the official file boundary and report the missing evidence.
Never fabricate calendar dates, a percentage, an embargo length, or a gap duration.

## Source and limits

The adopted protocol comes from the user's pasted text and the
[sggpls post of 5 March 2026](https://discord.com/channels/1408067327370596423/1408069208562208989/1479032289295077477).
That source describes the previous Predictorium competition. This ADR makes the
same protocol mandatory for our project; it does not retroactively establish
that current Connectome source timestamps have been independently verified.

## Consequences and supersession

- Supersedes prospective primary-selection fold defaults in AGENTS, ADR0031,
  ADR0043 and the October17 draft workflow where they conflict. Matched fitting
  seeds remain useful; random folds can only be labelled auxiliary diagnostics.
- Preserve all historical source pins, recipes, logs, predictions and scores.
  Do not reinterpret historical F0/F2 results as chronological confirmation.
- Frozen old checkouts remain replayable under their historical contract; do
  not modify a live or frozen recipe to insert this change.
- A final fit using validation requires a later explicit scope amendment and
  cannot report that same validation as independent performance.
- Model reconstruction, code preparation and unit tests do not authorize new
  scientific fits or restart any stopped GPU process.

## Verification

New admission checks must reject primary evaluation/fitting overlap, random-fold
membership masquerading as chronological, changed source identities and future
information in preprocessing/teacher ancestry. Synthetic checks establish code
behavior only. Actual source/data membership and temporal-order evidence remain
required at the receiving host; no data or training is accessed in this change.
