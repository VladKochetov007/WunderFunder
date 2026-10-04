# Preserved E465 OneCycle reference-check failure

This directory preserves the sealed v1 attempt and its exact source, draft,
sealed spec, runner attempt, log, and quarantine record. The worker exited
before importing Torch because the check required exact equality between the
port's first cosine value and the algebraically equivalent `0.001 / 25.0`.
That assertion was stricter than floating-point cosine evaluation. The
OneCycle implementation is unchanged in v2; boundary checks now use the same
predeclared LR/beta tolerances as the all-index PyTorch comparison. See
`preserved_manifest.json` for hashes of the retained files.
