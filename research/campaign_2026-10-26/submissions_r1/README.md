# Muon submissions — 2026-10-26 R1

Pre-declared [selection rule](SELECTION_RULE_R1.json); gate passed (stage-6 Muon − AdamW +0.00026).
Built by [build_packages.sh](build_packages.sh) with the [packaging tool](../packaging_r1/README.md).
ZIPs live in `submissions/muon_20261026/` (git-ignored; hashes in [SHA256SUMS](SHA256SUMS)). No upload.

| Package | Members (fixed coef) | Bytes | ABI / A–B–A reset | Callback wall, 64 seq | Panel F0 / F2 (descriptive) |
|---|---|---:|---|---:|---|
| `muon_single_gru_s3.zip` | `vt_all_h390m_s3_r1@ep5` 1.0 | 4,646,591 | pass / bitwise | 20.2 s | 0.660149 / 0.710516 |
| `muon_tl3_s3.zip` | h390m s3 .34 + FTT tlead s5 .33 + s6 .33 | 12,538,124 | pass / bitwise | 107.0 s | 0.654316 / 0.711899 |
| pair variants | s3+s4 | pending s4 fit | | | |

References on the same panel (Oct-24, AdamW students): single H390 S3 0.658380 / 0.710090;
TL3+H390 S3 0.653724 / 0.711724 (public-uploaded TL3 family ≈ .6683).

Native fidelity of the Muon member ([export log](export_h390m_s3.log)): native int8 vs torch FP32
RMS diff 0.56% of prediction RMS (H390 reference 0.72%), panel WP native − torch +0.00007 / −0.00017.

**Evidence limits.** All-data members (GRU and FTT) were fitted on every panel sequence: panel
scores verify the package and are not held-out quality. Muon's support: fast-recipe cross-fold
+0.0038/+0.0043 (local) and +0.0036/+0.0033 and +0.0041/+0.0033 (Kaggle), stage-6 transfer
+0.00026 [−0.00041, +0.00101]. Expected public change vs the H390-based ZIPs: small; no forecast.
Local callback time ratio suggests the single GRU runs ~5× faster than the TL3 blend (TL3 public
runtime 39:12); not a competition timing measurement.
