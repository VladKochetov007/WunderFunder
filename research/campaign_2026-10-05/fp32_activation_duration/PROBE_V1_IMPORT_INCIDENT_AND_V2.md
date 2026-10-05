# E489 probe v1 import incident and prepared v2 entrypoint

The sealed v1 CUDA probe failed during Python imports, before CUDA setup and
before any cohort/data read. The bootstrap added `activation_optimizer` to
`sys.path`, but imported `fit_activation_placement` without adding its package
directory. Root's preserved v1 runner log identifies the missing placement
module path. This is an execution/import failure, not an E489 scientific
result. The sealed v1 source and its attempted receipt remain unchanged.

A separate v2 probe entrypoint is prepared at
`gpu_probe_fp32_duration_v2.py`. Its only functional repair is to add
`research/campaign_2026-10-05/activation_placement` to the import path before
the placement-module import. It records its own versioned source filename in
the probe source inventory. No training, data read, GPU operation, retry, or
score was run while preparing v2.

The bounded check used the actual `scripts/limited` wrapper with one CPU,
2 GiB hard memory cap, and swap disabled. `py_compile` and Ruff pass; `--help`
imports the v2 entrypoint and exits zero before data/GPU code. Probe v2 SHA-256:
`7ee1bf79dac158d7aaa6530f55173126ad5e8f8df1c4e50832d83e9b0ca19b20`.

Root also recorded that v1's controller admission timestamp was 07:00:32.527,
after the 07:00 admission cutoff. There was no admission for the six full
trajectories. The prepared repair is for review and a later window; it is not
authorization to retry or a result about model quality.
