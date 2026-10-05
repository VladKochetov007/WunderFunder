# E485 probe v1 review incident

The independently reviewed probe source is preserved at
`GPU_PROBE_REVIEWED_V1.py` with SHA-256
`c0ab570cdd95dfdf2fc975460d98198227120042ecb7b79779c6668aea89a888`. The
working `gpu_probe.py` is now a separately versioned v2 candidate and has
different bytes. Neither source has been launched.

Review found that the native/custom Muon post-step comparison checked absolute
parameter values with `atol=1e-4, rtol=1e-3`. Those gates can pass or fail based
on parameter magnitude and do not directly bound the difference between the
two updates. The next version must retain the existing forward/state and
gradient tolerances, compare per-family parameter **update deltas** against
registered absolute and relative-to-reference-update-RMS gates, and include a
separate exact same-gradient test showing the local clipped update helper
matches frozen E474's helper on cloned models with a clipping-active gradient.

The critic recommended a 10% RMS update-error limit relative to each native
GRU-matrix or AdamW-family update RMS, plus a per-tensor max-error cap at half
the reference family/matrix update RMS with a `1e-7` absolute floor. These
thresholds and an exact shared-gradient local-vs-E474 helper test are recorded
in `GPU_PROBE_UPDATE_GATES.md` before probe-v2 edits. Do not widen a threshold
after an observed failure. No CUDA probe, full fit, or test score has been run
for E485.
