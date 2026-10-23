# R4 run logs

This folder holds the complete stdout/stderr log for each matched control and
feature fit, prediction export and official scoring command. Logs are tied to
the stable run ID, fold, seed and arm; the append-only `progress.jsonl` and
`resources.jsonl` in the parent folder record phase boundaries and measured
resource peaks.

The short profile logs live under `../profile/` and are not scientific fits.
Training logs report loss, learning rate, gradient diagnostics, update counts
and timing. A log alone does not establish a completed endpoint; see the
checkpoint provenance, prediction sidecar and official score receipt for each
run.
