# Research reset: fixes, measured EDA, 25 proposals

2026-10-17. The user requested extensive investigation and bug repair, explicitly
without model training. [CONTRACT.json](CONTRACT.json) permits bounded inference
of frozen checkpoints, descriptive analysis and code checks; fitting a calibrator,
linear probe or test-time optimizer is also excluded. Zero competition-data fits, GPU training
hours, paid resources and submissions were admitted. The required code suite
includes tiny synthetic optimizer/resume fixtures; these are disclosed engineering
checks, not research fits or trained competition candidates.

- [Findings and decisions](REPORT.md) — what changed, what the measurements
  establish, what remains unresolved, and which mechanisms deserve attention.
- [25 hypothesis cards](IDEAS_25.md), with [structured records](IDEAS_25.json) —
  exactly 25 diverse proposals, each with prior overlap, counterevidence,
  discriminating control, no-fit checks and a CPU/student path. Priorities are
  qualitative information value, not expected leaderboard gains.
- [Proposal draft packets](drafts/README.md) — 25 staged English PR bodies and
  explicit non-runnable recipes; implementation and fit admission remain pending.
- [Future comparison protocol](PROTOCOL.json) — 30 paired fold/seed cells per arm,
  strict FP32 and frozen estimand; model/data bindings and authorization remain
  pending. This is not an executable training recipe.
- [Bugfix packet](bugfix/README.md) — four reviewed server patches, hash-checked
  deployment with backups, seven regression tests and a four-sequence frozen
  checkpoint replay. The replay isolates a preprocessing failure, not a new model.
- [Measured EDA](eda/README.md) — all-sequence four-price-code event census,
  256-sequence full-feature sample and 64-sequence input lag/shift controls, with
  executed scripts, provenance, statistics and scope limits.
- [Sources](sources/README.md) — two independent searches plus coordinator checks:
  32 distinct primary literature works and live official rules. Abstract-level
  checks are distinguished from method inspection; reported gains do not transfer
  automatically to this competition.
- [Release checks](RELEASE_CHECKS.json) — independent hash review,66passed/2skipped
  code suite, disclosed synthetic optimizer fixtures and sparse-catalog limitation.
- [Independent reviews](reviews/README.md) — proposals, cross-criticism, evidence
  and metric audits, final portfolio critique, and recorded coordinator decisions.

Every existing validation panel is exposed. No new holdout, full-panel score
improvement, calibrated public-score forecast or top-1 guarantee is claimed.
The [unified index](../../unified/README.md) remains the navigation entry point.
