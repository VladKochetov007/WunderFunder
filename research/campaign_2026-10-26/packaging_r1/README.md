# Packaging tool — 2026-10-26 R1

[package_tool.py](package_tool.py): `export` (checkpoint → native member npz), `build` (TL3 engine
ZIP with replaced members, deterministic timestamps, ≤20 MB), `replay` (Python 3.11 native callback
on the frozen Oct-24 64-sequence F0/F2 panel, ABI + CRC + A–B–A bitwise reset), `score` (official
metric on that panel, optional agreement with a torch FP32 export).

Self-test (2026-10-26):
- `export vt_all_h390_ds_homogeneous_s3_r3 5` reproduces the packaged H390 S3 member byte-for-byte
  (SHA-256 `62b11349…`).
- Rebuilt TL3+H390 S3 replays **bitwise equal** to the Oct-24 quota S3 replay; panel WP 0.653724 / 0.711724
  (matches the quota grade).
- Single-member H390 S3 ZIP (4,631,474 bytes): native int8 vs torch FP32 export, F0 0.658380 vs 0.658402,
  F2 0.710090 vs 0.710185; RMS diff 0.72% of prediction RMS.

The panel is exposed and all-data members were fitted on it: scores are technical/descriptive only.
