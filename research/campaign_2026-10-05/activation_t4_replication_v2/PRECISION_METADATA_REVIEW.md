# E482 v2 precision receipt review

Review is limited to the frozen E482 v2 source and the actual provider receipt metadata. It does not inspect evaluation targets or compute scores.

The precision fields in the JSON receipts are snapshots taken after the precision contexts have restored their previous backend settings. In `native_parity`, the model forward/backward and optimizer-step comparison are inside `e474.full_ieee_float32()`, but the returned `precision` dictionary is obtained after that context exits. `run_probe` calls `native_parity` before entering its own fit-probe context, and likewise serializes its own `precision` field after exiting. `run_fits` serializes its field after the fit context. Therefore neither the nested native-parity snapshot nor the probe/fit top-level snapshot can establish the precision flags that were active during computation. The observed default values (`none`/`tf32`) in those fields are expected restoration values, not evidence of a failed guarded run.

The pinned worker source places the baseline/native parity calculation and all probe updates inside `full_ieee_float32()` contexts. The full-fit loop is also enclosed by that context. The actual GPU native parity receipt reports output, state, input-gradient, parameter-gradient, and clipped optimizer-step parity passed for each optimizer. This execution evidence supports the intended recurrence/gradient comparison at the checked shapes and tolerances. It is not a general proof of bitwise equivalence.

The precision wrapper does not make every optimizer operation IEEE FP32: E481's unchanged Muon orthogonalization path uses its existing BF16 Newton–Schulz calculation on CUDA. We therefore describe this as IEEE FP32 model forward/backward with the registered optimizer implementation, not “all-FP32 training.”

The independent grader must not gate or report the restored precision snapshots as if they measured active flags. It instead pins the worker and helper source hashes, requires the actual T4 probe and native parity receipts to pass, and verifies fit/probe continuity and completed updates. This note does not authorize evaluation scoring or claim a public metric result.
