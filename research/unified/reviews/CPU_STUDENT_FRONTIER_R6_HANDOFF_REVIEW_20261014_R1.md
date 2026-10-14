# F0 student-frontier R6 cache-handoff command review

**Date:** 2026-10-14  
**Reviewer:** independent critic  
**Scope:** read-only review of the seven R6 sequence plans and the R5 user-bus diagnosis. No handoff, cache reclaim, fit, export, grade, or provider action was run.

## Sources and provenance

The reviewed plans are in `research/campaign_2026-10-14/cpu_student_frontier_r1/recipes/CPU_STUDENT_FRONTIER_F0_S3_H*_*_R6_SEQUENCE.json` in MAIN. Their hashes are pinned by `R6_COMMAND_REPAIR_PREFLIGHT_R1.json` (SHA-256 `e01d66ef0fbaf6c5f259193f45647ab5a77d4946fef8956e88dab9acbf5bdcc8`). Each R5 base plan is in the frozen `student-frontier-ops-r5-20261014` worktree and its hash is recorded by that receipt. The unchanged helper `scripts/reclaim_student_frontier_cache.py` has SHA-256 `6b776d6b324b3fa6fa8b69fd4d9beb567ed76d83b2fa2d07144d0d44123e7ccc`.

`HANDOFF_USER_BUS_DIAGNOSIS_R1.json` (SHA-256 `3c5a4f19392f85da6e79760b7433cc2d3f73a2012f80f4d52c5d15421febd948`) records that the sanitized child failed to connect to the user bus before reclaim, while the same-user query with `/run/user/1000` bus variables succeeded and returned the campaign slice's 18 GB high, 20 GB maximum, and zero-swap settings. The probe did not invoke reclaim.

## Checks and finding

I independently hashed all seven R6 plans and all seven R5 base plans against the receipt. I then reversed only the declared plan changes: the R6 label/repair metadata, the two bus environment assignments at each of the two handoff worker commands, the fresh handoff attempt IDs, and the corresponding `_r6_after_` result paths. Each normalized R6 plan reconstructed its R5 base exactly. The other eight steps, including all fit, validation, profile, export, and grading commands, are byte-for-byte equivalent as parsed JSON. The handoff helper source is unchanged.

**Verdict:** bounded GO for root to admit the H256×1 R6 handoff attempt under a fresh resource check. The command repair is limited to supplying `XDG_RUNTIME_DIR=/run/user/1000` and `DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus` to the child that calls `systemctl --user`; the failed R5 attempt remains separate and preserved. The review does not show that Linux will reclaim the requested file-cache amount, establish a retained-cache floor, or admit the other six plans without their own resource checks.
