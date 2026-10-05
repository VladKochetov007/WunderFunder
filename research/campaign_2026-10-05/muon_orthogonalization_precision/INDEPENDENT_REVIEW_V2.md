# Independent E487 source review, v2 packet

Reviewed final v2 candidate identities:

- `precision_replay.py`: `480f2ae6d3aa09a3f972a3ad9fc682e7a92e07843d6e002434b2130ec75415d2`
- `gpu_probe.py`: `fc0a4c55a2adc33f9dbead86717e3a8ae9ef4d5fedf983c77a18bb9b83aa0f27`
- `tests.py`: `edcd15c558af0a2623c6a5727fd6b5c759d1d5044208148f2c076643e8b7ac5a`
- `cpu_preflight.py`: `6bcdbbc44c2d329c4c15c1b1ef3e3bb77594c6237fd62393e092e6937ab8f2d5`
- fit-only CPU v6 receipt: `2c7d2568b3a5e601b330af32cbcea0b70b8306503b5624720b99937d261a9437`
- v2 draft resource spec: `64d7581041610ad78898afa0db1eb8e7bd374c8f239661a4f89a5fb674b5a7a2`

This is a source review plus bounded CPU synthetic validation only. No CUDA
probe, full fit, E486/E485 evaluation, or test-role read was performed.

The v1 blockers are corrected. The fit loader now separately reads the raw
feature columns for all 24 fit groups, checks shape and finiteness before the
E470 `causal_features()` sanitizer, and records per-group raw-feature hashes.
The probe sets `WNN_GPU_MEMORY_BYTES=2147483648` before CUDA availability and
`configure_torch()`, so the registered allocator limit is active before model
allocations. Replay checks populated optimizer tensors and numeric scalars for
finiteness, and the aggregate finite flag is derived from the four replay
receipts and their updated states rather than hard-coded.

The precision comparison remains correctly isolated. The local FP32
orthogonalizer preserves the source quintic coefficients, normalization,
epsilon, transpose convention, five iterations, and final cast. The
`FP32MuonAdamW` step preserves the source momentum, decay, block scaling, and
AdamW branches. The BF16 arm delegates to the unchanged production
`MuonAdamW`. CPU tests verify exact CPU helper equivalence, exact same-gradient
clone states/deltas, frozen E474 helper equality for BF16, clipping activation,
strict-gate failure recording without early termination, raw-feature
nonfinite rejection, and optimizer-state nonfinite rejection.

The v2 draft pins 600 seconds, two CPUs, 6 GiB host RAM, 2 GiB allocator
memory, and a 3,000,000,000-byte observed-device stop. Its output directory is
v2-specific; the probe itself rejects an existing output path. The measured
v6 preflight uses only fit-role groups `[1062, 1725, 1214, 1766]`, binds the
same source hashes, and reports the 24 raw-fit-group checks. No data-dependent
strict-gate result is inferred from that CPU receipt: CPU source orthogonalizer
dispatch is FP32 for both arms.

Bounded validation of the exact current source passed:

```text
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  ../../../scripts/limited ../../../.venv/bin/python -m unittest -v tests
Ran 7 tests in 0.868s — OK

../../../scripts/limited ../../../.venv/bin/ruff check --output-format concise \
  precision_replay.py gpu_probe.py cpu_preflight.py tests.py — All checks passed
../../../scripts/limited ../../../.venv/bin/python -m py_compile \
  precision_replay.py gpu_probe.py cpu_preflight.py tests.py — exit 0
```

**Disposition: GO for the single registered E487 CUDA diagnostic probe after
the controller seals the v2 spec and checks current source/input hashes.**
No full fit follows automatically. Preserve every finite crossed replay and
all strict gate outcomes; a diagnostic pass remains one-step attribution and
cannot reopen the E485 failure or establish a training benefit.
