# Treatment kernel R6 deadline-wait repair

R6 is an identified successor to the frozen R5 kernel. It changes one runtime
detail in `kernel_r6/kernel.py`: the parent now waits for the full positive
floating-point wall time remaining before the 10,000-second kernel deadline.
The inherited `WNN_KERNEL_DEADLINE_MONOTONIC` value is unchanged. The bundled
`kaggle_gpu_runner.py` still resolves each worker timeout as the lesser of its
declared cap and the remaining kernel time minus the 300-second finalization
reserve.

This separates the worker watchdog cutoff from the parent subprocess timeout.
The runner therefore retains its cleanup interval to stop the worker process
group and write `failure.json` and `summary.json`. A worker timeout or other
failure remains a failed stage; declared partial checkpoints/artifacts are
preserved in the runner's failure receipt for terminal collection. The repair
does not alter science, worker argv, source/data identities, device count,
10,000-second wall limit, 20,000 GPU-second two-device ceiling, or reserve.

`kernel_r6/deadline_fixture_r6.py` drives the actual local
`scripts/kaggle_gpu_runner.py` timeout-resolution and watchdog/receipt code
with a fake process, fake device inventory and accelerated monotonic clock.
It verifies that the outer wait includes the full remaining wall time, the
worker cutoff precedes it by at least 300 seconds, the runner invokes its
process-group stop path, and a declared partial checkpoint is included in
`failure.json` before `summary.json` is written. This is an interface/cleanup
fixture only; it is not a Kaggle profile, GPU run, fit, or provider test.

R5 remains unchanged. R6 is prepared but unuploaded. Root admission, fresh
quota/session checks, READY input checks and independent R6 review remain
separate launch conditions.
