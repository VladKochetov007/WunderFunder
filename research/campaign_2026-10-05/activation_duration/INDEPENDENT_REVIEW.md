# E485 independent implementation review

Reviewed the v3 source packet on 2026-10-05. Current identities:

- `fit_duration.py`: `d9a1b9d243a9c9e50082fe0d9e7d98186d63cb0c566fcae0a20c00ccd42c4b48`
- `tests.py`: `4f0579a73a7492ce960582d7f744f9b29152e7b5860219593c9fc63a89005534`
- `gpu_probe.py`: `7f3b9a808eb8adbd0856b61fe8bd36ba53df2c1cd1455b697eb748b880d444fd`
- `cpu_preflight_v7/fit_manifest.json`: `bf23c9bdfb27a3f9367e8d305bcfa632bf410e6c0990fa2670e6538062bf3d8f`
- update gate registration: `GPU_PROBE_UPDATE_GATES.md`, `52d547abd2da11cd712e5973b2ef1ac2eca70b449cbd143abf8141bea221d164`

I independently reran the five synthetic tests under `scripts/limited` with a one-CPU, 2-GiB cap: all passed in 1.964 seconds. Ruff and `py_compile` also passed. No CUDA probe, full fit, or test-role data was run or read by this reviewer.

The implementation matches the registered reset-after GRU equation, preserving PyTorch's reset/update/candidate gate order and the hidden candidate bias inside the reset multiplication. It retains native parameter names and the `[2, batch, 128]` state, changes only the bounded candidate `u + alpha*sin(pi*u)`, and initializes both arms from byte-identical H128 native states for each paired seed. The 474-step epoch count is `79 chunks × 6 batches`; the four trajectories per shape are not restarted at snapshot boundaries. State carries and detaches across chunks within each four-sequence batch, while optimizer moments continue over all four epochs. Canonical cohort order is shuffled with `default_rng(seed + epoch_index)` and shared between paired shapes.

The optimizer grouping covers exactly the four recurrent weight matrices with three Muon gate blocks; biases, input projection, and head use AdamW. Source checks reject omissions, overlaps, and duplicate parameter assignment. The CUDA probe records active IEEE settings inside the context, identifies the existing BF16 Muon orthogonalizer, checks device memory after helper parity and after each native/custom parity execution before proceeding to fit updates, then checks it after every real update. The measured time projection uses the slowest actual-shape B4×256 update and must remain under the predeclared 7,200-second ceiling.

The v3 probe now compares update deltas from identical pre-step parameter snapshots and reports per-matrix and per-family RMS error and maximum absolute error. It separately applies identical clipping-active synthetic gradients through the local helper and frozen E474 update helper and requires byte-exact parameters and optimizer state. These checks close the v1 parameter-value-tolerance flaw. The source-level approximate native/custom delta limit is 10%; before admitting full trajectories, the controller must additionally verify the probe's emitted records satisfy the stricter predeclared gate: each matrix/AdamW family RMS error is at most `0.02 * reference_update_rms + 1e-10`, and every parameter tensor maximum absolute error is at most `2e-5`. This review does not treat a source-level `passed` flag as evidence that the stricter observed gate passed.

**Disposition: GO for the separately admitted actual-shape CUDA probe only.** Full-fit admission remains conditional on the measured v3 probe meeting both update-error gates, finite changed updates in all six seed/shape cells, memory and teardown receipts, and a runtime projection within the existing budget. This supports only a finite-training mechanism check on the exposed cohort, not convergence or public-score claims.
