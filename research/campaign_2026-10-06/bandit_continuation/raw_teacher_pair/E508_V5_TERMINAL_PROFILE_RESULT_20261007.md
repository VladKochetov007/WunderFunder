# E-508 v5 terminal profile result

E-508-v5 reached terminal `KernelWorkerStatus.ERROR` after the profile-only
worker completed its actual-device timing gate. This is an execution-feasibility
failure, not a teacher-quality result: `PROFILE_ONLY` prevented either teacher
fit or export from starting, and no score was calculated.

The corrected `--device cuda` selector passed the earlier packet review. Input
identity checks ran against the mounted private datasets; the profile log
records 10,607 train sequences, 1,405 eligible/used valid-train sequences, 468
evaluation sequences excluded, and 29,840 planned steps for the pinned recipe.
The measured short profile had `seq_per_s=14.98` and `step_wall=1067.84 ms`;
the worker projects with this value as milliseconds per step. It estimated
8,524 seconds for both fits and exports, above the frozen 6,300-second
production cap, and exited at the gate. The log also records `loader_wait` of
1,251.06 ms in the same profile units, so data-input latency should be isolated
in any new feasibility packet.

Kaggle output collection succeeded: 57 files were downloaded, and their hashes
are recorded in
[`e508_profile_terminal_v5.json`](../../cloud_preflight/e508_profile_terminal_v5.json).
The complete raw files remain in
[`downloads/e508_v5`](../../cloud_preflight/downloads/e508_v5/). No OOM was
reported, and the screen admitted no fit. The result therefore contributes
execution/profile evidence only and no numerical bandit update.

Do not blindly resubmit this packet. Any next attempt needs a new reviewed
recipe that addresses the measured wall-time bottleneck and rechecks whether
reduced compute changes the teacher comparison. A useful repair should first
separate data-pipeline time from recurrent compute instead of shrinking model
capacity by assumption.
