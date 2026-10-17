# Measured EDA, no predictive fitting

Read [FINDINGS.md](FINDINGS.md) for interpretation and explicit sampling limits.
All data remains on beleriand; files here contain scripts and aggregate/per-sequence
statistics, not the supplied feature matrices. Every panel remains exposed.

- [stream_audit.py](stream_audit.py) and [summary.json](summary.json): exact
  executed full112-feature stratified audit, 256 complete sequences/5.12M rows,
  seed20261017, 32 sequences per retrospective class per split. Contains event
  rates, target magnitudes, lag co-counts, full-row repetition and train-anchored
  marginal histograms. Equal-class selection is not natural pooled composition.
- [event_exceptions.json](event_exceptions.json): 52 explicit sampled exceptions
  to the old universal i0-AND target-clock law, checked as need-eligible and
  tick-sized. The census below gives total counts; the sample does not describe
  every exception's magnitude.
- [full_event_audit.py](full_event_audit.py),
  [full_event_summary.json](full_event_summary.json),
  [full_event_audit.json.gz](full_event_audit.json.gz) and
  [FULL_SCAN_RECEIPT.json](FULL_SCAN_RECEIPT.json): exact executed census of
  12,480 row groups/249.6M rows, four price-code columns plus labels/masks/IDs.
  Counts 1,253 need-eligible target changes outside the selected i0 conjunction,
  92 scored valid exceptions, and weight/age summaries. Compressed JSON preserves
  all per-sequence records; receipt binds raw/compressed/source SHA256 and private
  remote location. The uncompressed local working copy is ignored by Git.
- [event_lag_controls.py](event_lag_controls.py),
  [EVENT_LAG_CONTROLS.json](EVENT_LAG_CONTROLS.json) and
  [EVENT_LAG_CONTROLS.md](EVENT_LAG_CONTROLS.md): executed input-only control,
  64 full sequences/1.28M rows. Uses first8 RG-sorted members of each existing
  32-sequence stratum, two halves, lags1/4/16/64 and within-half circular shift1000.
  Shows persistence and heterogeneous i1 lead association; establishes no target
  gain or causality. The code reads no labels, but inherited strata are retrospective.

Executed scripts are frozen as run. No format-only rewrite changes their bound
hashes. Census coverage does not mean all112features were read for every row.
Train descriptive selection is `need_prediction`, valid is
`need_prediction & is_scored`. One row group per sequence is the supplied storage
contract; scripts assert sequence length but do not revalidate every row's ID.

Related ledger records: E-RR17-STRATIFIED-EDA, E-RR17-EVENT-CENSUS,
E-RR17-EVENT-LAG-CONTROL. H-001 has an appended correction; original evidence
remains intact. [Parent](../README.md).
