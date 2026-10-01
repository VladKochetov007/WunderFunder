The exact final student ZIP passed the complete 39.4M-callback runtime check in **2672.212 seconds (44.537 minutes)**, with 927.788 seconds of margin to the predefined 60-minute limit. This was one attempt, exit 0, without a timeout or retry.

| Check | Observed result |
|---|---:|
| Input callbacks | 39,400,000 |
| Emitted predictions | 39,204,970 |
| Warmup callbacks returning None | 195,030 |
| Complete sequences | 1,970 |
| Unique raw sequences | 1,873 |
| Additional complete repeats | 97 (1,940,000 callbacks) |
| Repeat outputs | All 97 emitted float32 hashes exactly match their original sequences |
| Fixed members / coefficients | 4 / 0.25 each throughout |
| Peak observed RSS | 356,604 KiB (348.246 MiB) |
| Process ru_maxrss | 355,640 KiB |
| CPU / memory limits | CPU 39, 100% quota; 16 GiB; swap 0 |
| Natural native path | v3 / VNNI INT8 |

The full-process time includes initialization, raw input I/O, the callback trace, validation and integrity checks. Callback loops themselves took 2548.282 seconds. The SMT sibling was busy for 8.999% of measured callback windows; quiet-runtime status is false. The elapsed time is the observed value, with no adjustment or contention-based retry.

The ZIP SHA256 is `eec571629b0829aeb9c73db80a1170fcb7cb3ccfd132f72f3ddd51f779d2fdd4` (6,000,321 bytes). Every ZIP entry matches its frozen manifest. Frozen source, plan, allocation, environment and checkpoint hashes match. The worker's complete raw-input hashes before and after agree, and the final raw-input stat still matches the plan. Detailed hashes and all 97 repeat comparisons are preserved in `outcome.json`.

The runtime scope is inactive and empty. All three observed process identities—outer timeout PID 260211/startticks 502953621, launcher PID 260212/startticks 502953625, and worker PID 260213/startticks 502953630—are absent. No runtime resource reservation remains.

This qualifies the specified complete workload on this host and its natural VNNI path. It adds no fresh predictive validation or timing guarantee for another host/ISA. No targets or scoring masks were decoded, and no WP query was performed.

`sync-manifest.json` records the final local/SSH file-hash comparison. The package is `package/student_fixed4.zip`; primary execution receipts are `execution/launch-result.json`, `execution/worker-result.json` and `execution/sequences.jsonl`.
