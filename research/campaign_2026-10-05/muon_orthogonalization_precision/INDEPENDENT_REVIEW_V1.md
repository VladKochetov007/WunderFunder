# Independent E487 source review, v1 packet

Reviewed source hashes:

- `precision_replay.py`: `4f5a3bf0733e3106a5743840e7e97706536962539f840747a958054275e0d726`
- `tests.py`: `813c6a63370222fd7c5db3bcd6626c2685c6d3c7eafd621fccbbeaa957a5d964`
- `cpu_preflight.py`: `6bcdbbc44c2d329c4c15c1b1ef3e3bb77594c6237fd62393e092e6937ab8f2d5`
- `gpu_probe.py`: `db8932ea07115d76472349444334d4e9873f8efd00915e2d8dee06019f9d11bb`
- `cpu_preflight_v5/preflight.json`: `9efdc271de63925318768df32cc6535dab667998cecddb7193b09e122325a35c`

Review was source-only apart from the bounded CPU synthetic test suite. No GPU
was used and no test-role data was loaded.

The numerical design is implemented coherently. The FP32 optimizer subclass
retains the frozen Muon/AdamW update equations and changes only the
orthogonalizer intermediate dtype; the BF16 arm dispatches through the frozen
optimizer. The same fixed-gradient replay is cloned across precision arms,
gradient clipping hashes are compared, and the crossed 2x2 replay records all
finite expected strict-gate failures instead of stopping at the first one.
The shared-gradient control uses non-collinear synthetic gradients, checks
clipping, exact same-device clone deltas/state, and the frozen E474 BF16 helper.
The FP32 CUDA/CPU calculation uses the registered elementwise tolerances, and
the result records per-matrix/family errors and stores the crossed and shared
delta arrays. Resource checks are placed before and after every replay.

The bounded synthetic test command passed:

```text
WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G WNN_RUNTIME_MAX=300 \
  ../../../scripts/limited ../../../.venv/bin/python -m unittest -v tests
Ran 5 tests in 0.826s — OK
```

There are two source blockers before CUDA admission:

1. `capture_actual_fit_batch()` checks finiteness only after calling
   `e470.load_cohort(False)`. That loader passes raw parquet features through
   `causal_features()`, whose `np.nan_to_num` can replace NaN/Inf values before
   E487 sees them. Labels are explicitly validated before clipping, but the
   raw feature finite check is not. Add a fit-only raw-column check before
   feature construction, and bind that check in the source receipt.
2. `replay()` checks gradients, deltas and updated parameters for finiteness,
   but does not check optimizer-state tensors. Since
   `replay_crossed_gradients()` reports `all_crossed_finite=True`, explicitly
   finite-check each optimizer-state tensor before recording that status.

Resource note: a direct invocation without `WNN_GPU_MEMORY_BYTES` would use
the shared default 70% allocator fraction, and an observed-peak check alone
does not cap allocation. The sealed campaign runner injects
`WNN_GPU_MEMORY_BYTES=2147483648` for a 2-GiB GPU spec, so this is not a claim
that the admitted runner would use the default. Setting the cap explicitly in
the probe remains a useful defensive check.

**Disposition: NO-GO for CUDA on this packet.** The two source fixes need a
versioned update and fresh source hashes/preflight. Do not reinterpret the
existing CPU pass as evidence for CUDA precision or relax any registered
parity/update gate. The E485 failed probe and its no-fit-admission status
remain unchanged.
