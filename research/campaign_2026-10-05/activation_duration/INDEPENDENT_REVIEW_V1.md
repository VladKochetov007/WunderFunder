# E485 source review, probe v1

Reviewed the frozen source packet on 2026-10-05. Hashes observed:

- `fit_duration.py`: `9d1d24d0aa9dd53bd442deac7ec8534392ae64eb22161e7efa1e0de00dcb7b30`
- `tests.py`: `44b72a15d4e479d2509cc20fb9bbbe4ce140d28b8a7da7368ffb2290425dd979`
- `gpu_probe.py`: `c0ab570cdd95dfdf2fc975460d98198227120042ecb7b79779c6668aea89a888`
- `cpu_preflight_v5/fit_manifest.json`: `8e55dee2d67515fc54102d78a48b7dfc45468de5353e094d9ec319270bfd3e25`

The independent bounded CPU run of `tests.py` passed all four tests in 1.931 s under the local 1-CPU/2-GiB guard. Static review confirms the registered six fit-only recipes, 474 updates per epoch, four continuous epochs with optimizer state preserved, state carry/detach within each four-sequence batch, canonical shared row ordering across paired shapes, exact four-matrix/three-block Muon partition, and active-precision capture inside the IEEE context. The model keeps the native H128 state/key layout and modifies only the bounded candidate formula. No CUDA probe, full fit, or test-role data was run or read by this reviewer.

## Probe v1 disposition: NO-GO

The v1 CUDA parity path compares post-step **parameter values** using `atol=1e-4, rtol=1e-3`. That absolute tolerance is not tied to the magnitude of the Muon update and can mask a materially different update when a recurrent matrix's step RMS is on the order of or below `1e-4`. The code does not report the discrepancy relative to the reference update. This is a weaker gate than the forward/gradient evidence warrants and must be corrected before any CUDA probe.

For the corrected probe, compare deltas from the identical pre-step snapshots. Require per-parameter maximum absolute delta error at most `2e-5`, and for each optimizer family require `RMS(delta_candidate - delta_native) <= 0.02 * RMS(delta_native) + 1e-10`; record the actual numerator, reference RMS, and ratio. This tests update fidelity directly and prevents tolerance from depending on parameter magnitudes. Add a separate exact shared-gradient/global-clipping optimizer-helper parity check so it independently verifies the local and frozen E474 MuonAdamW helper path. Keep the existing native forward and gradient tolerances as approximate parity checks; do not use their result to relax the update-delta bounds.

The observed concern is limited to the probe's update-equivalence gate; it does not invalidate the frozen training recipe or the CPU preflight. This review does not admit a CUDA probe or full training run.
