# Q8/I16 input projection precompute experiment

The isolated native change precomputes `x[j] * column_scale[j]` once per member-row and reuses it across that projection's output tiles. It keeps the existing ascending nonzero-column order and the existing separate floating-point multiply/add accumulation in each tile. No shared nonzero scan, GRU fusion, YMM rewrite, fast-math, or model/weight change was included.

The projection-level Rust tests passed bitwise comparisons against the original I16 AVX2 and Q8 VNNI paths over ragged dimensions, including the six-tile Q8 shape. The bounded library test run passed 55 tests with one pre-existing ignored test. The exact-package Python 3.11 panels used the frozen two-member 50/50 Q8 package and the same ten complete 20,000-row sequences, resetting before each sequence and each of two passes. All 400,000 callbacks per panel were finite, replay output was bitwise stable, and the per-sequence prediction SHA maps were identical between baseline and candidate in both paired rounds.

The alternating callback timing did not establish a repeatable gain:

| Paired round | Order | Baseline (µs/callback row) | Candidate (µs/callback row) | Candidate change |
|---|---|---:|---:|---:|
| 1 | baseline, candidate | 100.0416 | 100.3866 | 0.345% slower |
| 2 | candidate, baseline | 102.5228 | 100.1799 | 2.285% faster |

The two paired deltas have opposite signs. Their unweighted aggregate is about 0.986% faster for the candidate, but with only two pairs and within-panel repeat variation, that aggregate is not evidence of a stable speedup. The performance gate therefore failed; keep the baseline package as the supported choice and do not upload the candidate ZIP.

Panels ran under a one-CPU cgroup on an Intel i9-11900H, not the target AMD EPYC 9224. There was no material quota throttling: three panels recorded zero throttled periods, and one candidate panel recorded one period / 61 µs during an approximately 43-second run. The benchmark's process `ru_maxrss` field reports 1,368,028 KiB for every panel, while the same scope's cgroup `memory.peak` is 230,572,032–231,370,752 bytes. These measurements disagree and are preserved as a telemetry limitation rather than conflated.

The benchmark-only candidate package is 12,937,243 bytes (23,224 bytes larger than the 12,914,019-byte baseline), below the 20 MB limit. Its ZIP SHA-256 is `93daff3a56b5d10a42bc1172ea5e3dd6815a98f1998c3b6094bbcb9cfc989383`; the baseline ZIP SHA-256 is `d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284`. The candidate was exercised successfully on local Python 3.11, but Kaggle/bookworm ABI compatibility was not separately promoted or claimed after the timing gate failed.

`RESULTS_R1.json` contains the per-panel receipt pins and resource deltas. `CANDIDATE_BUILD_RECEIPT_R1.json` preserves the exact build receipt (SHA-256 `f0badac8aa20285a3cde8424e76594da5e00b8a31ea567af81429c517b54168c`). `IMPLEMENTATION_DIFF_R1.patch` preserves the exact source patch; its SHA-256 is `4bb9f41a0d71bf930d2290e8ca95832d32222cee5a8d42e255bde0ef2f34f8f3`. The native implementation is isolated in commit `2b339ab1093b50a08154c68542a89dd8b3b0e5e6`; it should not be cherry-picked into the active submission source. The original optimization record and build receipt are pre-panel snapshots; this report and results receipt provide the later decision without rewriting them.
