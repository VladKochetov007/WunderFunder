# W384 serial v2 failure

The enabled background watcher observed ERROR and saved provider logs. Root collected the terminal diagnostic files into diagnostics/wnn-night-w384-serial-20261005-v2/.

The CUDA preflight failed before any scientific fit: 24 legacy actual-trainer resume/export fixtures failed, 12 passed. The new bounded reader requires 20,000 rows, while these fixtures deliberately create eight-row groups. The runtime cause is `ValueError: row group 1 has 8 rows`, rather than OOM. Final visible cgroup memory.events has zero OOM and kill counters. Outer process-tree peak RSS was 5,850,218,496 bytes; observed body duration 477.228 seconds. Preserve this failure and source.

A mitigation must maintain strict production data identity and row/target/mask consistency, restore genuine synthetic-fixture coverage, and pass independent transport plus trainer tests before a separately reserved retry. No tests may be broadly skipped to make the gate pass. This failure says nothing about the scientific width hypothesis.
