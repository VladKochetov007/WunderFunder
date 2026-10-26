# Muon submissions — 2026-10-26 R1

Pre-declared [selection rule](SELECTION_RULE_R1.json); gate passed (stage-6 Muon − AdamW +0.00026).
Built by [build_packages.sh](build_packages.sh) with the [packaging tool](../packaging_r1/README.md).
ZIPs live in `submissions/muon_20261026/` (git-ignored; hashes in [SHA256SUMS](SHA256SUMS)). No upload.

| Package | Members (fixed coef) | Bytes | ABI / A–B–A reset | Callback wall, 64 seq | Panel F0 / F2 (descriptive) |
|---|---|---:|---|---:|---|
| `muon_single_gru_s3.zip` | `vt_all_h390m_s3_r1@ep5` 1.0 | 4,646,591 | pass / bitwise | 20.2 s | 0.660149 / 0.710516 |
| `muon_tl3_s3.zip` | h390m s3 .34 + FTT tlead s5 .33 + s6 .33 | 12,538,124 | pass / bitwise | 107.0 s | 0.654316 / 0.711899 |
| `muon_gru_pair_s3s4.zip` | h390m s3 .5 + s4 .5 | 8,270,699 | pass / bitwise | 49.1 s | 0.659705 / 0.710921 |
| `muon_tl3_s3s4.zip` | h390m s3 .17 + s4 .17 + FTT tlead s5 .33 + s6 .33 | 16,162,230 | pass / bitwise | 114.5 s | 0.654039 / 0.711834 |

References on the same panel (Oct-24, AdamW students): single H390 S3 0.658380 / 0.710090;
TL3+H390 S3 0.653724 / 0.711724; TL3+H390 S3/S4 0.653954 / 0.711958 (public-uploaded TL3 family ≈ .6683).

Native fidelity of the Muon member ([export log](export_h390m_s3.log)): native int8 vs torch FP32
RMS diff 0.56% of prediction RMS (H390 reference 0.72%), panel WP native − torch +0.00007 / −0.00017.

**Evidence limits.** All-data members (GRU and FTT) were fitted on every panel sequence: panel
scores verify the package and are not held-out quality. Muon's support: fast-recipe cross-fold
+0.0038/+0.0043 (local) and +0.0036/+0.0033 and +0.0041/+0.0033 (Kaggle), stage-6 transfer
+0.00026 [−0.00041, +0.00101]. Expected public change vs the H390-based ZIPs: small; no forecast.
Local callback time ratio suggests the single GRU runs ~5× faster than the TL3 blend (TL3 public
runtime 39:12); not a competition timing measurement.

## How different is the Muon student? (full valid exports, all scored rows, descriptive)

| Pair | corr t0 | corr t1 |
|---|---:|---:|
| Muon h390m s3 vs AdamW H390 S3 (same parent) | 0.99953 | 0.99955 |
| AdamW H390 S3 vs S4 (different seed/parent) | 0.99373 | 0.99416 |
| Muon h390m s3 vs AdamW H390 S4 | 0.99365 | 0.99405 |

The 6-epoch distill stage moves the warm Muon-trained parent very little; optimizer choice there
changes predictions ~12× less than the seed/parent does. The Muon ZIPs are therefore near-copies
of the Oct-24 H390 quota ZIPs; expect public differences within noise. **Seed/parent diversity
(s3+s4 pair) is a larger lever than the optimizer in this lineage.**

## Recommendation (for the human upload decision)

1. **Best blend: `muon_tl3_s3s4.zip`** — TL3 structure, Muon student pair in the student slot
   (mirrors the Oct-24 `tl3_h390_s3_s4` rule; seed-pair diversity is the larger lever).
   Runtime above TL3 (one extra GRU member, local callback 114.5 s vs 107.0 s for 64 seq; the
   AdamW sibling was built under the same budget). Fallback/cheaper: `muon_tl3_s3.zip`.
2. **Best single GRU: `muon_single_gru_s3.zip`** (4.6 MB, ~5× faster than TL3 locally).
   `muon_gru_pair_s3s4.zip` is the GRU-only two-seed alternative.

Honest expectation: the Muon students correlate 0.9995 with their AdamW siblings, so these ZIPs
are expected to score within public noise (sd ≈ 0.0008) of the Oct-24 H390 quota ZIPs. They are
the strict-FP32 Muon versions of our strongest lineage, not a new information source.
