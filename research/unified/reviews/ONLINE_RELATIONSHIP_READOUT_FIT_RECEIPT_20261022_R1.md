# Online i1-to-i0 readout fitter implementation receipt

This code-only receipt covers the fixed three-epoch online relationship residual-head fitter in [`online_relationship_fit.py`](../../../src/wnn/training/online_relationship_fit.py) and its contract checks in [`test_online_relationship_fit.py`](../../../tests/test_online_relationship_fit.py). The implementation reuses the canonical residual head, event features, primary pair WMSE, fixed-checkpoint selector, and compiled causal relationship stream. It forms 419 inputs from the parent-normalized 211-column base and 208 clipped relationship-state columns; it retains row order across 8,192-row batches and their final partial tail. Evaluation labels and row weights are not read.

The component was committed as `8418117a0b198b683cf28d3d99cfb8ab1ea60fbb`; the batch-test snapshot correction is `bd9d6a12683363d4313bd0aff19a84c83a112867`. Static Ruff checks passed. The training-structure check had passed for the component before the test-only correction; the fitter remains 299 lines.

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

Delivery confirmed that its in-progress cache iterator yields a four-field sequence with matching attribute names: `raw`, `target`, `row_weight`, and `parent_prediction`; the fitter's evaluation path uses only `raw` and `parent_prediction`. Its sidecar will bind ordered source references. This was an interface review, not a validation of the unfinished cache writer or CLI. No model fit, accelerator profile, full data-cache read, grade, or quality claim was made.
