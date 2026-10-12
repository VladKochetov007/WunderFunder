# Fixed clean-T3 CP5 + old-Z4 half-blend screen

This preregistered screen tested one root-frozen candidate: 50% of the clean-T3 F0 control student at CP5/epoch 5 plus 50% of the old Z4 F0 student at CP1/epoch 1. The only comparator was the exact old Z4 endpoint by itself. It used the existing official fold-0 panel scorer on the two already-materialized prediction banks; it did not run inference or training.

On the previously inspected F0 excluded-468 panel, the fixed blend scored **0.68325495** and the old endpoint scored **0.68314181**, for a paired delta of **+0.00011314**. The paired whole-component bootstrap interval was **[-0.00128701, +0.00151656]** (`n=141` components, 800 replicates, seed 271828). This is a small, uncertain conditional comparison, not evidence that the new recipe wins or should be promoted.

The comparison is bound to the exact checkpoint, prediction, sidecar, provenance, split and row-group fingerprints in `RESULT_HANDOFF_R1.json` and the scorer result `GRADE_FIXED_HALF_R1.json`. Both prediction arrays have shape `(1873, 20000, 2)`, are stored as float32, and cover canonical valid-source row groups 0–1872. The new endpoint used FP32 inference; the old endpoint used FP16 inference. Their teacher/initializer ancestry and recipes also differ, so this is not an isolated architecture or precision effect.

The evaluation units are 468 groups from F0, an exposed development panel that has been repeatedly inspected. The interval reflects resampling of its 141 connected components only. It says nothing about seed-to-seed, recipe, precision, other-fold or public-transfer uncertainty. No public score was estimated, and the result does not authorize a submission decision.

The preregistration was committed before scoring (`60a9c58f3f3190a94291d70e2c8160ac9e4e0b22`; JSON SHA-256 `b868ef8f3ef7a50a14c57b8dbeddf6a64f90a3c7bb2f880bc67d8746e4fdfbe6`). The bounded score command completed in 6.64 seconds with exit 0, 1,402,112 KiB maximum RSS, and no swap under a 1,500,000,000-byte support cap. The aggregate admission values and memory-event attribution caveat are recorded in the handoff. The cgroup `memory.max` event counter rose by 7 while another fit worker was active; OOM counters stayed unchanged, and the transient screen scope's exact PID/peak was not retained.

Files:

- `PREREGISTRATION_R1.json` and `.md`: immutable screen definition.
- `GRADE_FIXED_HALF_R1.json`: official metric and paired component-bootstrap result.
- `RUN_STDOUT_R1.txt` and `RUN_STDERR_TIME_R1.txt`: raw command output and `/usr/bin/time -v` record.
- `RESULT_HANDOFF_R1.json`: endpoint/source hashes, execution/resource record, result and interpretation limits.
