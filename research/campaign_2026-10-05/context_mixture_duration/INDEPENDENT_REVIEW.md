# Independent source review: E488 duration study

Review date: 2026-10-05 06:29 UTC. This is a source and fit-only audit; no
CUDA probe, full trajectory, or evaluation scoring was run.

## Reviewed identities

- Design: `E488_DESIGN.md`, SHA-256
  `3764a095504af6da66a4faa908a47ff5dd3f94266249d5bea7c07cf611148824`.
- Blind critique: `SPEC_CRITIC.md`, SHA-256
  `ccc5caba6eef86b928e31cc78ce65602ad528a6549b0017a3d91ffcf1a3aa3e1`.
- Controller clarifications: SHA-256
  `d382f867bd8c889c5c63f40228ba1133e0f52d2a9e396860fbf15009fced0a9c`.
- Trainer `fit_duration.py`: SHA-256
  `89d38cd71da3ff8905d51da60d369bbe04b07d26789ab242085d59d384ac299e`.
- Tests `tests_duration.py`: SHA-256
  `29a2b247d5f32e32c5242325c0bbce35f511a44fc32d01c429d29f252d33413d`.
- Probe `gpu_probe_duration.py`: SHA-256
  `3e1f383c9364e70b2ea0e4846a15de5281090326a1d5fe634061dbaebea5747f`.
- Fit-only preflight manifest: SHA-256
  `d4e25b8e8e84a7506ec5a8eecf973791b3f2aa07569bf5c64704970f6ea8fae6`.

## Source findings

The trainer's registered trajectory structure is consistent with the design:
three seeds crossed with four arms; three uninterrupted epochs per trajectory;
474 B4×256 chunk updates per epoch; AdamW moments are retained; sequence state
is detached between chunks and reset at each batch; canonical fit positions
are independently shuffled with `default_rng(seed + epoch_index)` for the
zero-based epoch index. It records all six specified global steps, snapshots
epochs 1 and 3, and checks the optimizer step counter at every epoch boundary.
The common seed initialization is copied into each arm and the tests check the
three mixture initial functions, no extra gate RNG draw, and two-step parity
with E474's frozen clipping/AdamW helper. The fit loader requests only the
24-role fit cohort and validates the expected array shapes and normalization.

The bounded CPU checks passed independently under `scripts/limited` with one
CPU and a 2 GiB memory cap: all five synthetic tests passed; Ruff and
`py_compile` passed. The recorded preflight manifest reports 12 unique
seed/arm cells, finite losses, changed parameters, `[2, 4, 64]` recurrent
states, and `test_rows_loaded=false`.

## Probe admission blocker

**NO-GO for the current CUDA probe source.** Full training in
`fit_duration.py` runs inside `e474.full_ieee_float32()`. In contrast,
`gpu_probe_duration.run_probe()` invokes both `e484.check_reference_parity()`
and all 12 actual-shape candidate updates without that context. Its checks
therefore do not validate candidate updates under the precision policy used by
the admitted full fits. The report also does not capture backend precision
metadata while that policy is active. This is an implementation mismatch, not
a request to loosen tolerances.

Before a probe, either version the probe to wrap the parity and all 12 cells in
the same `e474.full_ieee_float32()` context used by training and save precision
metadata from inside it, or provide a separately sealed wrapper whose code
and executed identity prove it encloses the complete `run_probe()` call. The
review must bind the exact resulting source/wrapper hashes. Keep the original
probe source immutable as the historical reviewed version.

The probe records allocator allocated/reserved and peak values after parity
and each update, but does not itself enforce the 2 GiB allocator or 3 GB
physical-device stop. Admission therefore also requires the sealed runner to
set `WNN_GPU_MEMORY_BYTES=2147483648` before CUDA configuration and enforce the
3 GB device stop; reject the probe if its receipt exceeds either bound. The
900-second ceiling remains a separate controller gate: record probe wall
time, use measured matching H64/B4×256 throughput with initialization,
checkpoint and teardown margin, and admit the complete 12×3-epoch matrix only
if that projection fits. No full-run admission is implied by this source
review.

No actor source was edited by this reviewer. No GPU, full fit, or test-row
loader was invoked.

## Probe-wrapper v2 addendum

The controller supplied a separate wrapper after the probe-source blocker.
`gpu_probe_duration_v2.py` SHA-256 is
`d7abd232ebbec9a10ec5dff2415103ff75ce672633e512034426669347489d7b`. It
enters `e474.full_ieee_float32()` before `run_probe()`, records active
precision metadata from inside the context, and keeps the parity call and all
12 actual-shape updates within that context. It records synchronized per-cell
update wall times and computes a deliberately conservative full-matrix
projection as the slowest measured update × 1,422 × 12 × 1.5, plus measured
probe time and a 120-second fixed overhead allowance. Ruff and `py_compile`
pass for this exact wrapper. It leaves the reviewed v1 probe untouched.

**Conditional GO for one actual-shape probe using this exact v2 wrapper**,
provided the sealed runner pins this hash, injects the registered 2 GiB
allocator limit before CUDA configuration, enforces the 3 GB physical-device
stop and 900-second cap, and treats a projection over 900 seconds as a
fail-closed rejection before any full trajectory. The wrapper records allocator
allocated/reserved snapshots; the external monitor remains responsible for
whole-device usage. This disposition does not admit full trajectories in
advance: the actual probe must pass all 12 cells, report active IEEE precision,
finish without OOM and teardown cleanly, and produce a measured projection
within the registered cap.
