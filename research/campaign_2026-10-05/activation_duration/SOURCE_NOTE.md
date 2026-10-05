# E485 source packet

The actor is `fit_duration.py`; `tests.py` contains synthetic checks and
`gpu_probe.py` is an actual-shape CUDA probe source. The probe has **not** been
launched. No E485 full trajectory or test scoring has run.

The actor retains E478's two-layer reset-after GRU scan and native parameter
keys, scales it to H128, and changes only the bounded candidate to
`tanh(v) + alpha*sin(pi*tanh(v))`, with fixed alpha 0 or 0.1. It uses the
frozen E470 fit-only loader/loss, E474 full-IEEE context and telemetry
arithmetic, and existing MuonAdamW implementation. The four recurrent matrices
alone receive Muon (three gate blocks, momentum 0.9); AdamW uses betas
`(0.9,0.999)` for all other parameters. The existing CUDA Muon orthogonalizer
intentionally uses BF16; “full IEEE” applies to model forward/backward math,
not that optimizer kernel.

The run is six continuous trajectories. There are 474 updates in each epoch
(79 chunks × 6 batches), 1,896 per trajectory, and 11,376 across the matrix.
Optimizer buffers continue across all four epochs. Per epoch, row-role indices
are shuffled in canonical cohort order with
`default_rng(seed + epoch_index).shuffle(np.arange(24))`, shared by both
paired shapes. State resets between four-sequence batches and epochs and is
carried/detached across each batch's chunks. Model checkpoints at epochs 1, 2,
and 4 are 18 snapshots from those six fits, not independent fit arms.

`cpu_preflight_v7/fit_manifest.json` records six actual fit-only B4×256
updates, one for every seed/shape cell. All six had finite loss, gradients and
parameters, and changed parameters. The same four fit row groups are used for
both shapes within each seed. The preflight reports the registered 474 updates
per epoch and pins the actor, tests, probe and numeric update-gate hashes, but does not run full
trajectories.

Earlier preflight directories are preserved. `cpu_preflight_v1` was
superseded because an erroneous metadata formula used `78 chunks × batch size
4 = 312` rather than `79 chunks × 6 batches = 474`; its six one-step probes
were not full fits. `cpu_preflight_v2` corrected the count but predates the
final explicit optimizer-source and active-precision metadata bindings.
`cpu_preflight_v3` had correct counters and metadata but predates the probe
source binding. `cpu_preflight_v4` binds the probe source before the final
byte-exact device-initialization check. `cpu_preflight_v5` predates the
per-family update-delta gates and exact shared-gradient helper parity test.
Use only v7 as the current CPU source packet. v6 binds the earlier v2 GPU
probe and remains preserved.

The unlaunched v3 probe retains the registered native-versus-custom
alpha-zero output/state and gradient tolerances, then compares post-step
deltas by recurrent matrix and AdamW family under critic-agreed absolute and
relative-to-update-RMS gates. A separate clipping-active same-gradient test
requires byte-exact local-vs-frozen-E474 parameters and optimizer state,
including CUDA BF16 orthogonalization. On actual fit-only B4×256 data it also
runs one finite, changed update for all six cells, verifies initial CPU bytes
after device transfer, captures active backend flags, and enforces 2 GiB
allocator / 3 GB observed-device stop limits. Its projection uses the slowest
measured warm-up step across cells with a 25% margin; it excludes
initialization, checkpointing, evaluation and scoring and is an admission aid,
not a fit-time guarantee.

`versions/gpu_probe_v2.py` preserves the previously reviewed probe. The v3
probe adds allocator/observed-device checks immediately after shared-gradient
helper parity and after each native/custom parity cell, before it enters the
six real fit-only update probes. Its own early-stop update-delta thresholds
remain the preregistered 10% relative RMS and 0.5× reference RMS per-tensor
maximum. The root admission reviewer separately applies the stricter
preregistered 2% RMS and 2e-5 maximum-absolute delta gates to the emitted
per-matrix/family measurements before any full trajectory is admitted.
