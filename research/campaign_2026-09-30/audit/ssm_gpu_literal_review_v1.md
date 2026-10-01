# Independent GPU-to-literal CPU replay

**FAIL for multi; PASS for single**, at the preexisting mixed bound `abs(error) <= 2e-5 + 2e-4*abs(CPU literal)`. No threshold was changed. Final synthetic student weights were loaded strictly and matched the profiler checkpoint byte-for-byte at every tensor. Frozen model SHA256 remains `227a5b8fe8ada7b34d61463ad2f022a0d802011aa985b215ef880d5a469c1756`.

The saved reset batch1/length2000 FP32 input was evaluated with the previously independently written scalar/real2x2 sequential recurrence. The GPU multi output passes (maximum absolute difference4.787e-6); state0 passes (maximum1.9002e-4), but state1 has two mixed-bound violations (maximum4.8542e-4, worst bound ratio1.09664). CPU parallel evaluation versus the same literal recurrence passes, including state1 maximum1.5259e-5. Single passes every readout/state comparison, with state maximum1.1206e-5.

The saved GPU flags match fresh inherited defaults: highest matmul precision, CUDA matmul TF32 disabled, cuDNN TF32 enabled and deterministic. This measurement does not establish a cause for the device discrepancy. Multi remains blocked on this frozen deployment parity gate until a documented correction or explicit preregistered scientific disposition; numerical tolerances are not silently relaxed.

`ssm_gpu_literal_replay_v1.json` preserves all raw differences, hashes and flags. Durable job `c20260930_independent_ssm_gpu_literal_v2` exited1 to indicate the numerical failure; replay0.463s, observed peak552.05MiB, nested hard1core/2GiB/swap0 verified, CUDA uninitialized. The preceding v1 job failed while loading the author's inert `TorchVersion` string subclass under `weights_only=True`; its job/log is retained. The successful loader used a narrowly scoped safe-global for that class and still used `weights_only=True`.

No competition data or targets were read. This checks final synthetic student/reset FP32 only, without establishing teacher, full trained checkpoint, AMP or quality behavior.
