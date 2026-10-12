# Exact activation/absmax fusion, frozen runtime comparison R1

The candidate is bitwise identical to the baseline over all ten sequences in all four panels. The predefined runtime gate fails: the candidate is faster in the first paired comparison and slower in the second. This is mixed, effectively tied local evidence, so stop H-CPU-MAX-01's activation/absmax fusion treatment here. Keep the patch and candidate as isolated benchmark evidence; do not promote the package.

Each panel replayed ten complete 20,000-row feature sequences twice, for 400,000 callbacks, with reset before every sequence and pass. All reset checks passed, all required outputs were finite, all four panels selected native v3/Q8VNNI, and per-sequence prediction SHA-256 values matched across baseline and candidate.

| Alternating pair | Baseline mean callback time | Candidate mean callback time | Candidate speed change |
|---|---:|---:|---:|
| 1: baseline, candidate | 100.613500 µs/row | 99.374378 µs/row | 1.231566% faster |
| 2: candidate, baseline | 99.407516 µs/row | 100.602192 µs/row | 1.201797% slower |

The sign reversal fails the frozen repeatability criterion. No additional benchmark or tuning was run. The comparison measures only the shipped callback on one local Intel Core i9-11900H with CPU 3 affinity; it does not certify AMD EPYC 9224 portal runtime or model quality.

The four runner receipts record the feature-panel SHA, package and build-receipt hashes, per-sequence feature/prediction hashes, CPU/platform/Python/ABI selection, callback timing, and per-scope cgroup CPU/memory observations. Each scope had one CPU quota, a 1,199,996,928-byte memory maximum, zero swap, and no OOM/max events; observed cgroup peak was approximately 231–232 MB. The candidate package builder had already verified ZIP CRC and that `v3/wnn_native.abi3.so` was the only changed member. The runner imported the exact candidate ZIP and confirmed v3/Q8VNNI, though this bounded panel is not a full submission streaming gate.

No metric, local score, public score, or submission-quality claim is made. Full pins, protocols, raw receipts, and computed values are in `PREBENCHMARK_FREEZE_R1.json`, `CANDIDATE_BUILD_RECEIPT_R1.json`, and `RESULTS_R1.json`.
