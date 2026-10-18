# PR15 geometry/discrete B300 review

**Disposition:** source-level CPU gates pass on the full deployed source, including Torch CPU parity (6 passed in 3.98 seconds; receipt `/data/epanov_s/tabdata/receipts/pr15-cpu-tests-v2.log`). Conditional go for the new PR15 cohort remains subject to the two-worker-per-card density profile passing alongside healthy R01 workers. The feature implementation is causal and the four controls are width-matched. I found no feature-level reason to block the first fit. This is not a claim that PR15 improves prediction quality; the full preregistered matrix and grading still decide that. No full PR15 fits have started; I ran no SSH commands or GPU work.

## Source and recipe

The locally reviewed PR source package is commit `e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e`, archive SHA256 `29fb624f73a4d6ca8dc38a428fb73fa7fec2c410617b18781c4e0a8d30506985`. Its 172 files match the package manifest by path, size, mode, and file hash. It contains source and experiment metadata, not market data, targets, caches, or pretrained weights. The deployed 172-file tree still needs to remain bound to this commit/package identity in each run receipt.

The arm mapping is `baseline=all`, `moments=moments_only`, `dictionary=dictionary_only`, and `combined=none`. Each arm retains the same 60 geometry/discrete columns and the same 271-wide total input (112 raw + 99 event + 60 derived). The exact masks are respectively all 60, columns 20–59, columns 0–19, and no mask. The recipe uses fresh H512 × 1 GRUs, B8, 2,000-row chunks, three epochs, fixed epoch 2, no evaluation during fit, train+valid fit membership, WMSE/AdamW/OneCycle, learning rate 0.001, weight decay 0.01, gradient clip 1.0, deterministic strict FP32. `common.train_command` binds the split manifest, fold, seed, control, and these settings; `pair_identity` plus the frozen matrix defines the 30 common fold/seed pairs for each of the four arms.

## Causality and controls

The transform uses only the selected raw channels `(0, 3, 52, 55)` at the current row and earlier rows within that sequence. Cumulative mean, standard deviation, residual, atom occupancy, normalized value, current category, category occupancy, dominant-event share, and dwell time reset at sequence boundaries. The dictionary/vocabulary and arc/atom prior are fitted from fit-membership raw sequences; labels and masks are not inputs to this feature fit. The trainer integration fits preprocessing on training batches, with online normalization disabled by default. The resulting checkpoint carries the prior and the selected control for inference. The full-sequence `predict_run` path rebuilds the same features from the checkpoint. `StreamingResearchFeatures` explicitly rejects this mode, so current evidence supports full-sequence prediction only; it does not establish stateful streaming-serving parity.

The audit-only CPU gates are in `context-followup/pr15_review_cpu_tests.py`. They check the NumPy transform against a separate row-wise oracle, suffix-mutation prefix invariance and per-sequence reset, exact 60-column mask partitions, all 120 matched arm/cell commands, profile attempt arithmetic, and Torch-vs-NumPy parity plus Torch prefix invariance. The corrected suite passed on the complete deployed source with Torch CPU enabled: 6 passed in 3.98 seconds, recorded at `/data/epanov_s/tabdata/receipts/pr15-cpu-tests-v2.log`.

The first full-source run exposed a test-harness call error: the Torch feature function's fourth positional parameter is the output buffer, while `control` is fifth. The test now passes `control=...` explicitly. This was a reviewer-test API mistake, not a production-source finding; the corrected Torch gate has since passed on the B300 host.

## Profile and launch binding

The frozen profile command is the canonical `common.train_command(profile_id, fold, seed, "none")` with `--profile-steps 8 --profile-warmup 2`. The CLI flags count batches, despite `PROFILE_CONFIG` calling them “updates.” With 20,000 rows per sequence and 2,000-row TBPTT chunks, each B8 batch makes 10 optimizer attempts: the 2 warmup plus 8 measured batches therefore mean 100 total attempts, of which 80 are timed. The launcher should check actual `PROFILE_UPDATE_EVIDENCE` reports 100 attempts and finite completed updates, along with strict FP32 evidence; do not treat the labels in the profile config as 10 optimizer steps.

The original `run_cell.py` must not be reused unchanged for this admission. It requires a clean Git checkout and exact project-root/path mounts, expects an eight-card profile receipt, schedules one fit per card, and does not bind CPU affinity or a safe per-process CUDA allocator limit. Those assumptions do not describe two new PR15 workers sharing each B300 with a live R01 worker. The bounded custom adapter should call the exact canonical builder and stages, not fork trainer or feature code:

```python
command = common.train_command(run_name, fold, seed, common.CONTROL_BY_ARM[arm])
subprocess.run(command, cwd=pr15_root, env=slot_environment, check=True)
```

For profile workers, use the same builder with control `"none"`, then append `("--profile-steps", "8", "--profile-warmup", "2")`. Give each process its unique run/profile name, dedicated output/cache/scratch roots, explicit source/runtime hashes, disjoint CPU affinity, and a conservative explicit `WNN_GPU_MEMORY_BYTES`; the default allocator fraction is based on each process's observed free memory and is unsafe as a shared-card reservation. Keep one process on each assigned GPU slot and verify aggregate memory, finite-update evidence, and R01 health before admitting the remaining cohort. After each fit, call the canonical `common.prediction_command(name, fold)` and `common.score_command(name, fold)` serially before releasing that slot; preserve distinct run names for every arm/fold/seed and require artifact plus score receipts.

The host-level cgroup/profile gates and the first real-shape concurrent profile are operational admission conditions, not feature review claims. Any profile instability or R01 degradation should pause the new cohort without touching R01 or foreign processes.
