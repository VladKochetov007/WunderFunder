# Local phase-wall utilization audit — 2026-10-08

Window: **2026-10-08 00:00–14:40:44 UTC** (52,844 seconds; 14.68 hours). This is a common timestamped subwindow, not the full campaign since Oct 6.

Successful terminal phases recorded **30,868.230 seconds** (8.57 h): **29,027.765 s fitting** across 10 runs, **80.777 s profiling** across 7 runs, and **1,759.688 s prediction/export** across 8 runs. Their interval union is **58.42%** of the wall window. Three failed/interrupted fit/export attempts account for another **1,554.685 process-wall seconds**, excluded from the successful numerator.

**58.42% is a recorded phase-process-wall share, not measured GPU utilization.** Attempt receipts have no consistent GPU-active timer; process elapsed includes model/input setup, export I/O, and batch/data loading during fits. Thus it is an upper-bound proxy for GPU-active time in the recorded successful windows. Strict loading/startup/idle-excluded compute seconds/share are unavailable. The ledger is partial across all local useful activity, so 58.42% is not an upper bound on campaign-wide useful work.

Successful plus failed/interrupted attempt windows cover **32,422.914 s** (61.36%). **20,421.086 s** have no eligible attempt in this ledger. Gaps can contain CPU analysis, grading, cache/data work, launch/queue activity, idle time, or missing records; they are not labeled idle.

| Phase | Successful attempts | Process wall seconds | Failed/interrupted attempts | Failed/interrupted seconds |
|---|---:|---:|---:|---:|
| Fit | 10 | 29027.765 | 2 | 1549.634 |
| Profile | 7 | 80.777 | 0 | 0.000 |
| Prediction/export | 8 | 1759.688 | 1 | 5.050 |

Largest unaccounted spans (no eligible phase attempt in the canonical ledger):
- 2026-10-08T06:16:03.120845+00:00 to 2026-10-08T07:18:41.883885+00:00: 3758.8 s
- 2026-10-08T13:54:38.412118+00:00 to 2026-10-08T14:40:44+00:00: 2765.6 s
- 2026-10-08T09:17:08.354650+00:00 to 2026-10-08T09:37:55.800466+00:00: 1247.4 s
- 2026-10-08T07:54:28.569254+00:00 to 2026-10-08T08:13:33.988947+00:00: 1145.4 s
- 2026-10-08T09:38:00.839382+00:00 to 2026-10-08T09:56:21.335205+00:00: 1100.5 s
- 2026-10-08T01:47:54.251530+00:00 to 2026-10-08T02:04:53.521834+00:00: 1019.3 s

The JSON lists each included attempt, source-receipt SHA-256, clipped interval and all unaccounted intervals. Provider time and CPU-only scoring/review are excluded. Oct 7 is omitted to keep the denominator in the common timestamped local-attempt window. No Torch import, model data read, or fit was performed.
