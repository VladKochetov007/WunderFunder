# Online i1-to-i0 readout fitter implementation receipt

This code-only receipt covers the fixed three-epoch online relationship residual-head fitter in [`online_relationship_fit.py`](../../../src/wnn/training/online_relationship_fit.py) and its contract checks in [`test_online_relationship_fit.py`](../../../tests/test_online_relationship_fit.py). The implementation reuses the canonical residual head, event features, primary pair WMSE, fixed-checkpoint selector, and compiled causal relationship stream. It forms 419 inputs from the parent-normalized 211-column base and 208 clipped relationship-state columns; it retains row order across 8,192-row batches and their final partial tail. Evaluation labels and row weights are not read.

The component was committed as `8418117a0b198b683cf28d3d99cfb8ab1ea60fbb`; the batch-test snapshot correction is `bd9d6a12683363d4313bd0aff19a84c83a112867`. Independent source review prompted a post-update parameter-finiteness guard and an evaluation-prediction finiteness guard in follow-up commit `f6a1ee193a816f9bc29e8d0935e6cdf521b4dc68`. Static Ruff and training-structure checks pass; the fitter is exactly 300 lines.

The first bounded test attempt passed two checks and failed the batch-order test because the test collected references to the generator's intentionally reused batch tensors. The test now snapshots each yielded batch, matching the fitter's immediate consume-before-resume behavior. The corrected run passed all three tests:

```text
3 passed in 5.30s
Elapsed wall time: 7.78s
User CPU: 3.50s; system CPU: 0.41s
Maximum RSS: 982,160 KiB
Swaps: 0
Exit status: 0
```

The run used a transient user systemd scope with `MemoryMax=1,100,000,000`, `MemorySwapMax=0`, `CPUQuota=50%`, one CPU affinity (`taskset -c 0`), `CUDA_VISIBLE_DEVICES` empty, and one OMP, MKL, OpenBLAS, and Numba thread. `TMPDIR` and `NUMBA_CACHE_DIR` pointed under `/home/vlad/development/wunder/research_tmp/online-residual-fit-component-20261022/tmp/` on the physical `/home` filesystem. Pytest used the repository's provisioned environment, explicit `PYTHONPATH=src`, and plugin autoload disabled.

The finite-guard follow-up injects a NaN into an optimizer-updated parameter and a separate NaN prediction from the evaluation head; both paths raise before state or prediction export. All five fitter contract tests passed in the bounded follow-up run:

```text
5 passed in 11.80s
Elapsed wall time: 17.21s
User CPU: 3.68s; system CPU: 0.62s
Maximum RSS: 958,484 KiB
Swaps: 0
Exit status: 0
```

The prior run annotation reported `MemoryMax=1,100,000,000`, `MemorySwapMax=0`, `CPUQuota=25%`, one CPU affinity, hidden GPU, and physical scratch/thread settings. The receipt does not preserve the exact launch command, transient scope identity, or terminal cgroup counters needed to verify those limits. Treat its resource-boundary claims as unverified; the recorded process-level `/usr/bin/time` measurements remain the available evidence.

Delivery confirmed that its in-progress cache iterator yields a four-field sequence with matching attribute names: `raw`, `target`, `row_weight`, and `parent_prediction`; the fitter's evaluation path uses only `raw` and `parent_prediction`. Its sidecar will bind ordered source references. This was an interface review, not a validation of the unfinished cache writer or CLI. No model fit, accelerator profile, full data-cache read, grade, or quality claim was made.
