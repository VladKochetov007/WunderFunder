# Model metadata mapping correction, R2

The first bounded model run is preserved in
`MODEL_RESULTS_R1_PRECORRECTION.json` (SHA-256
`69f3efc3cfbb4ed445ea26ebfe3747b6d7d76c885baa0a83b4d8c16a7f3a229f`). It used
the R1 code at commit `85cfa5c8c6fbab2708edfca2aafff06cf27ed8e1`; its code
SHA-256 was `cfc5c8068dbdf2470ec01b2e28d67deebc77c23d9805bd7bd9ea16abae6e7965`
and its contract SHA-256 was
`58a6e3381067cec3133d7f38456b2b4645551f9adf4292f3e66b38adf6f1d46b`.
The exact R1 contract and code remain recoverable from that commit. The R1
result also binds its input coverage file (SHA-256
`dbaacdade13daff25ce7a5bd86b8dcbca66849964df1ca483d47fc6cd8882b3b`), but it
does not record the execution command; no command is retroactively attributed
to that run. R2's exact command and input/script/contract hashes are frozen in
`MODEL_RUN_PLAN_R2.json` before its single run.

Two deterministic mapping defects were found after the frozen R1 run.

1. R1 omitted the alias-only, source-cited `c20261008_h230_*` rows from the
   existing `h230_hft_student` family. The frozen contract requires using an
   audited alias when source identifiers are absent.
2. R1 concatenated the package alias even when source/member identifiers were
   present. This violated source-first precedence. For `0catwwxg`, the exact
   source-only text maps to `legacy_since_ema`; adding its `chunk500` alias
   incorrectly mapped it to `chunked_ensemble`.

R2 applies both corrections in one identified mapping-only successor. The
feature list, target rows, penalty, source-component construction, CV split
rules, and baselines remain fixed. The initial result is retained; because it
was inspected before these corrections, the R2 table is a post-hoc
sensitivity/corrected retrospective diagnostic, not a fresh confirmatory
evaluation. Neither result is a current-candidate forecast.
