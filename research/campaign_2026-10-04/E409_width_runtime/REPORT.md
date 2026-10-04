# E409: measured D256 and W384 payload runtime

Completed successfully from 2026-10-04 01:33:09 to 01:38:05 UTC. This is a CPU
runtime and numerical-parity experiment, not a predictive-quality comparison.

The ordinary trained W384 checkpoint measured **401.617 thousand cycles per row
median**, range 392.027–403.171. Its median scorer-callback time was 107.733 us/row,
range 104.951–119.201. The review's unmeasured 500–540 kcycles estimate is not
supported for this one-head payload. W384 remains a runtime candidate. These
measurements do not establish a 52-minute or 60-minute scorer runtime; no old
cycles-to-public-minutes conversion was applied.

| Payload | kcycles/row min / median / max | us/row min / median / max |
|---|---|---|
| D256, trained ordinary head | 190.568 / 195.424 / 198.838 | 50.921 / 52.144 / 55.445 |
| D256, duplicated head sizing control | 221.148 / 222.711 / 224.257 | 57.927 / 59.493 / 64.507 |
| W384, trained ordinary head | 392.027 / 401.617 / 403.171 | 104.951 / 107.733 / 119.201 |
| W384, duplicated head sizing control | 444.261 / 452.945 / 466.915 | 119.830 / 122.069 / 138.641 |

The two-head controls duplicate the same trained head and average its outputs.
They measure head-width cost. They are not separately fitted heads, an improved
ensemble, or an actual two-member pair bag. Each ordinary model came from a real
checkpoint: `c20261001_final_d256_s3@ep3` and `c20261003_w384_s3_f0@ep3`.

Protocol: one pinned logical CPU (15) on Intel Core i9-11900H; three complete
20,000-row sequences from `cache/valid_subset20.parquet`; five repeated trials
with alternating case order. Every timing includes feature updates, native
callback, DataPoint construction and output validation. Model construction and
parquet reads occur outside the timed row loop. All raw trials, hardware text,
load averages, CPU affinity and hashes are retained in `result.json`,
`timing_trials.jsonl`, `manifest.json` and the sibling control directory.

Both v3 and v2 passed the predeclared gates: maximum prediction error <=3e-4,
absolute pooled WP difference <=5e-6, exact repeat and reset behavior. All four
q16x prediction arrays were bitwise identical between v2 and v3. W384's ordinary
head had max error 1.47045e-4 and pooled metric difference +4.86512e-7 versus
float32. These three-sequence scores are parity diagnostics, not scientific
validation. The native b2 source snapshot matches the seven current runtime
source files; the source and all three binary hashes are in the manifest.

Every stage used `scripts/limited` under `wnn-campaign.slice`, with CPU quota
100%, MemoryMax 2 GiB, swap zero, CUDA visibility disabled, and a 900-second
deadline. Maximum RSS was 667,052 KiB during export, 262,000 KiB for v3 parity,
261,932 KiB for v2 parity and 245,992 KiB for timing. All exit codes were zero.
The shared parent's OOM/oom_kill/oom_group_kill counters stayed zero. Its
hierarchical `max` counter increased from 44,225 to 92,260 during concurrent
campaign activity; this probe cannot attribute those memory-reclaim events.

Next deployment gate: package the exact surviving scientific predictor and
measure its end-to-end offline one-vCPU scorer-image runtime. The ordinary W384
shape is not ruled out by this CPU probe. Actual pair-bag runtime remains an
unmeasured separate question.
