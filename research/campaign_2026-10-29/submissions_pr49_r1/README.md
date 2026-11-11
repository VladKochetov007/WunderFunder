# PR49 TL3C submission ZIPs (October 29, R1)

User: "Give me submission zip path. do your best" and "We have optimized runtime exactly for cpu
model competition host uses. make sure to use it." Built with the PR51 packet (Ivanhacker13,
commit `677e40c48`) from Semyon's PR49 checkpoint release (`tabdata-pr49-checkpoints-20261029-r1`,
SHA-256 verified; split `0c4de3a2…`). No training, no blend-weight tuning (PR51's frozen
coefficients), no upload (human gate).

## Runtime

Team `wnn_cpp` (from `submissions/team_tl3_original_20261018.zip`, SHA `d80e69d3…`, accepted by
PR51), bit-identical to the binaries in our uploaded TL3 ZIPs (v3 SHA `3cb45462…`). Mode
`dynamic_int8`: dynamic INT8 + fast activations, `quant_backend="auto"` (Zen 4 AVX-512 VNNI on
the EPYC 9224 host) — the same settings as our previously uploaded TL3 ZIPs. Our Rust "D fused"
engine is not applicable: it does not implement the tiedlead/event encoders.

## Results (all 1,873 official valid sequences, 3,528,273 scored rows, official metric)

| ZIP | SHA-256 | Size | Served WP | Torch WP | µs/row (laptop) |
|---|---|---:|---:|---:|---:|
| `submissions/tl3c_pr49_20261029/blend-dynamic_int8/submission.zip` (0.34 fbc1k_s5 + 0.33 tlead_s5 + 0.33 tlead_s6) | `8f939726…` | 12.53 MB | **0.692336** | n/a | 116.8 |
| `submissions/tl3c_pr49_20261029/tiedlead-s5-dynamic_int8/submission.zip` | `c69055c2…` | 4.95 MB | **0.691434** | 0.691415 | 38.7 |

- Native INT8 single reproduces the PR49 Torch score (+0.00002): no feature/semantic drift
  through the CPU engine. A-B-A reset bitwise, warmup-state independence pass, 0 skipped rows
  for both ZIPs.
- Blend − single = **+0.0009**, above PR51's predeclared +0.0005 mixture threshold.
- Runtime: the original team TL3 ZIP (same three-member structure) measured 100.6 µs/row on this
  laptop (Oct 24) and completed in 39 min on the portal. The blend's 116.8 µs/row (measured at
  93 °C with concurrent containers) scales that to roughly 45 min on the portal — a projection,
  not a guarantee. The single ZIP projects to ~15–20 min.
- The blend container exited 124 (wrapper `timeout 3600` hit during post-loop hashing after a
  4,436 s loop, slowed by the 3-way CPU-0 restart incident); its report contains complete
  metrics and passed state checks. The f32 single check was stopped as redundant (INT8 already
  equals Torch) to reduce laptop heat.

## Limits

All models are train-only (ADR0044) and full valid has been inspected by teammates (exposed
selection evidence). Past local→public offsets were −0.006 to −0.010; no public score is
forecast. PR51's consumed-teacher-cache ancestry limitation applies unchanged.

## Incidents

PR51's validator pins itself to the first allowed CPU; three concurrent containers all pinned to
CPU 0 (33% each). Rootless podman lacks the cpuset controller and crun resets `taskset`
affinity, so `cpu_check.sh` sets affinity inside the container before running the validator
(`PIN` env). Docker daemon absent: rootless podman with `label=disable` (SELinux) used instead.
Aborted attempts preserved under `submissions/tl3c_pr49_20261029/aborted_*`.

Evidence here: manifest `inputs.json`, build receipts, full-valid reports, the check wrapper.
ZIPs and weights are git-ignored under `submissions/`.

## Public results (portal, uploaded by the user 2026-10-29; recorded 2026-11-11)

| ZIP | Served full valid | Public | Portal duration | Memory |
|---|---|---|---|---|
| blend-dynamic_int8 (`8f939726…`) | 0.692336 | **0.66355** | 41m57s | 656 MB |
| tiedlead-s5-dynamic_int8 (`c69055c2…`) | 0.691434 | **0.66246** | 18m6s | 629 MB |
| tiedlead-s5-f32 | (Torch 0.691415) | **SCORE:TIMEOUT** | 61m56s | 627 MB |

- Valid ordering transferred: blend − single = +0.0009 valid, +0.0011 public.
- Local→public offset −0.029 (train-only models; valid is a shifted block, see validation_r1).
- Both are **below** the full-corpus team TL3 ZIPs (tl3 0.66832, tl3_h390_s3_s4 0.66806), whose
  members also saw valid in fitting: train-only selection evidence does not survive without the
  final train+valid refit. Simon's full-corpus w18-final scored 0.66453 (Nov 4).
- The F32 single exceeded the portal time limit even though the INT8 blend of three members fit
  in 42 min: FP32 CPU GRU serving of this encoder family is not viable; use dynamic_int8.
- Blend-runtime projection (≈45 min) matched the portal (42 min).
