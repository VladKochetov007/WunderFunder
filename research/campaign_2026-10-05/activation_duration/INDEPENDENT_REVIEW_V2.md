# E485 probe v3 follow-up

The independent review of probe v2 confirmed the six-cell recipe, fit-only
preflight, state carry/reset behavior, per-matrix/family delta gates, and exact
shared-gradient helper parity. It identified one missing early resource check:
peak allocator and observed-device memory were not checked immediately after
native/custom parity allocations, before the six fit-update probes.

Probe v2 is preserved byte-for-byte at
`versions/gpu_probe_v2.py` (SHA256
`7fd7f567f88315301d94d6ed96778fed376f780fab9d0cb50abe7146ef313faf`). Probe
v3 now checks memory after the shared-gradient helper and each native/custom
parity cell. The fresh fit-only CPU preflight is `cpu_preflight_v7`; no CUDA
probe, full trajectory, or test-role read has occurred. The actor/tests remain
unchanged.

The independent reviewer also notes that the probe's registered 10% relative
RMS and 0.5× maximum-error bounds are its early-stop criteria. Root's separate
full-fit admission check is stricter: it must reject any emitted matrix/family
measurement exceeding 2% relative RMS or 2e-5 maximum absolute delta error.

Current hashes:

- `fit_duration.py`: `d9a1b9d243a9c9e50082fe0d9e7d98186d63cb0c566fcae0a20c00ccd42c4b48`
- `tests.py`: `4f0579a73a7492ce960582d7f744f9b29152e7b5860219593c9fc63a89005534`
- `gpu_probe.py` v3: `7f3b9a808eb8adbd0856b61fe8bd36ba53df2c1cd1455b697eb748b880d444fd`
- `cpu_preflight_v7/fit_manifest.json`: `bf23c9bdfb27a3f9367e8d305bcfa632bf410e6c0990fa2670e6538062bf3d8f`

The v3 actor-side validation passed all 5 synthetic tests, Ruff formatting and
lint, bytecode compilation, CLI import, and six actual fit-only B4×256 CPU
updates (all finite and changed). The exact revised probe is awaiting a quick
independent source recheck before any CUDA launch.
