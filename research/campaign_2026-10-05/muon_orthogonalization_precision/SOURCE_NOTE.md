# E487 source packet status

E487 is a one-batch diagnostic for the precision sensitivity described in
`E487_DESIGN.md`. It does not train a trajectory or use any test-role inputs.
The BF16 source path delegates to the frozen `wnn.optim.MuonAdamW`; the
FP32-only optimizer subclass is local to `precision_replay.py` and changes
only the intermediate dtype in the same five-step Newton–Schulz formula. No
production optimizer or E485 source was modified.

The diagnostic captures gradients once from the native GRU and custom
alpha-zero GRU on seed 271836's first canonical fit batch and replays each
captured set through both optimizer precisions from byte-identical parameters
and fresh zero momentum. Every finite crossed replay is retained even when
one or more native/custom strict gates fails. Gradient, clipping, parameter,
optimizer-state and initialization hashes accompany per-matrix/family update
RMS and maximum-error values. All four crossed delta arrays and five
same-gradient-control delta arrays (including the frozen E474 BF16 reference)
are written in `replay_deltas.npz`.

`cpu_preflight_v5/preflight.json` is the current bounded fit-only preflight.
It used only row groups `[1062, 1725, 1214, 1766]` from the fixed 24-fit-role
cohort, with batch size 4 and 256 rows. It passed five synthetic tests, the
native/custom forward and gradient parity checks, exact shared-gradient clone
checks, and BF16/frozen-E474 CPU parity. On CPU the source orthogonalizer uses
FP32, so BF16-source and diagnostic-FP32 update deltas are identical there;
the CPU result does not answer the CUDA precision question. Preflight v1/v2
are preserved; v5 binds the final probe status reporting with every crossed
gate result preserved, and checks before and after every CUDA replay.

The CUDA probe is source only and has not been launched. It compares synthetic
CPU/CUDA FP32 orthogonalizer outputs under the registered elementwise
`1e-5 + 1e-4*abs(cpu)` bound, captures active IEEE flags inside the context,
checks memory before/after every replay, and retains complete numerical
evidence. A source review from `resume_accelerators` is pending before CUDA
admission. Even an FP32 strict-gate pass would explain only a single zero-
momentum update; it does not admit E485 or establish a training benefit.
