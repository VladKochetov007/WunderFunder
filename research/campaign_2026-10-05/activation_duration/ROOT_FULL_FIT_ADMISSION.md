# E485 full-trajectory draft admission gate

This is a draft only. Do not seal or launch the six-trajectory spec until the
v3 probe completes successfully, its actual result artifact and execution
receipt are content-bound as inputs, and root verifies every strict condition
in `ROOT_GPU_PROBE_ADMISSION.md` (including the independent 2% relative RMS
and 2e-5 maximum update-delta gates). The probe must also fit the 7,200 second
projection cap. Root must confirm remaining local GPU budget, E483/E484 queue
priority, no competing GPU worker, and the existing 07:00 UTC admission cutoff.

The six-trajectory draft currently binds the immutable design, source, fit-only
preflight, dataset and cohort inputs, but it cannot bind the future probe
result before that result exists. Root must add the successful probe result and
execution receipt to the inputs and regenerate the content digest before any
seal or launch. Test-role data must remain unread during fitting.
