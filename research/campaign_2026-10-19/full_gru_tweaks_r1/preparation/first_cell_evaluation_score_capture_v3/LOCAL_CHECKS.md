# Local checks — score capture V3

The new focused fixture checks the V2-to-V3 AST delta, exact evaluation-v3 root/export-v3 unit, V5 scorer byte/SHA pins, exact systemd Id guard, strict native terminal-success tuple, synthetic admission/launch receipt joins, and the still-unbound actual score receipt literals. It does not run the capture reader's `main()` or query systemd, B, cgroups, GPU processes, or result files.

Ruff `check` and `format --check` are run only on the V3 template and its new focused fixture. No old reader fixture or suite is replayed. The root-side status/capture driver remains root-owned and will need its own versioned identity binding before any future capture.
