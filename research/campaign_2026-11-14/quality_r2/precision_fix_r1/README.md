# Correct the legacy Muon precision adapter

The [independent audit](../PRECISION_AUDIT_R1.md) found that the frozen PR49 legacy optimizer factory never forwarded `precision=fp32` into `MuonAdamW(strict_fp32=True)`. Saved optimizer groups from the interrupted q14 fit confirm `strict_fp32=False`. Model forward/backward was FP32 with TF32 off, but Newton–Schulz intermediates were BF16. Earlier quality measurements remain observations of that actual recipe; the all-math strict-FP32 description was wrong.

The [prospective amendment](../PRECISION_AMENDMENT_R2.json) stops the old conditional refit path, preserves the interrupted checkpoints and admits new matched controls. The interrupted fit consumed 30m40.938s wall / 35m35.452s CPU with 16 GiB cgroup memory peak ([unit receipt](interrupted_unit.log)); this cost is retained inside the original eight-local-GPU-hour cap. No fixed endpoint is claimed.

[Patch](strict_factory.patch), source commit `48776fba8af26b16b7e6813def7ccf571f56c1ac` on `experiment/q14-strict-muon-20261114`, adds the precision flag to both ordinary/grouped legacy optimizer constructors. It changes no objective, schedule, model, dataset, initializer or teacher. Optimizer switching is not enabled or covered by this patch. The frozen original source at `188c0788e` is untouched.

- [CUDA fixture](check_precision.py) exercises both factory branches: six optimizer updates and 24 actual Newton–Schulz calls with internal FP32 parameter/gradient/state/intermediate guards active. [Result](CUDA_CHECK_R1.json) passes on the actual RTX3060 Laptop / Torch2.14 runtime.
- The identical fixture rejects the original factory with `strict_fp32=False` ([negative control](negative_old_factory.log)). This is the bug-specific failure, not an unavailable-device or import failure.
- Ruff format/check pass. The repository structural checker passes 124 scoped training modules for both [base](structure_base.log) and [fixed](structure_fixed.log) sources. Its discovered scope does not include the preexisting larger legacy `training_dynamics.py`; no unrelated refactor was made.
- The full-recipe [engineering profile](../q14s_kd90_profile_s3_r2.fit.log) passes 120 finite completed updates, no scaler/skips, at 20.67 sequences/s. The first profile attempt was rejected before fitting because continuation-state saving conflicts with a profiling exit; R2 removes that flag for profiling only. Both logs remain preserved. Neither is a scientific quality fit.

Production fitting remains conditional on new complete paired results. Native export/parity checks and an honest all-data handoff are still required; the CUDA fixture does not establish quality or portability to an unobserved provider runtime.
