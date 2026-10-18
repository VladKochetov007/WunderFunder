# Actual preflight records

`preflight-r2-20261018t1520z` is the first actual CPU preflight attempt. The initial staging transport failed because gzip bytes on tar stdin lacked `--gzip`; extraction never began. `STAGE_FAILURE.json` and an independent empty-root observation are preserved. A separate exact recovery script added that transport flag, required the existing exact root to be empty/non-symlink, and retained create-only extraction. No frozen source, loader, manifest or plan changed. `STAGE.json` and the raw remote receipt confirm the recovered stage.

Root admission is `../ROOT_ADMISSION_20261018T1520Z.json`. The unit started at 15:20:53 UTC under 50% CPU / 1.5 GB / swap 0 / 900 s, CUDA hidden, physical scratch. Acceptance remains pending until the process and raw receipt are captured. This is source hashing and loader membership verification, not model export, fitting or quality evaluation.

The exact executed recovery script is archived beside its receipts. Raw CSV/SVG diagnostics elsewhere retain producer line endings/spacing because their SHA bindings are evidence; those generated files alone are excluded from whitespace lint, never rewritten.
