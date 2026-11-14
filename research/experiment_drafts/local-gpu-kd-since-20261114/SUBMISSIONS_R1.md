# November 14 submission handoff

**Later precision correction:** source review found FP32 model kernels with BF16 Muon Newton–Schulz in this original recipe. The all-math strict-FP32 description is superseded; numerical scores, hashes and native checks remain recorded observations. Corrected-source experiments are a separate panel.

The user requested current status and submissions from the strongest available variants, and supplied four portal receipts. All 18 queued fits completed before collection. This handoff uses existing fixed-epoch checkpoints; no new fitting or portal upload. Source: local fit logs, run provenance and exported predictions under `/home/vlad/development/wunder/research_tmp/chrono_root/`; package runtime is the SHA-pinned original team TL3 engine.

## Deliverables, in upload priority order

Files are in `/home/vlad/development/wunder/wnn_connectome_starterpack/submissions/local_20261114/`.

| Priority | ZIP | Training | Size | Evidence and use |
|---|---|---|---|---|
| 1 | `final_kd50_s3s4_pair.zip` | train + valid; seeds 3/4, equal weights | 8,257,503 bytes | Main new candidate; same fixed pair of train-only models gives valid 0.691423, +0.000949 over mean single. Final pair itself has no held-out score. |
| 2 | `final_kd50_s3_single.zip` | train + valid; seed 3 | 4,645,911 bytes | Faster fallback and direct all-data transfer comparison against uploaded train-only kd50 s3. This is the same already-built single ZIP, now with full Torch-export parity evidence. |
| 3, exploratory | `kd90_s3_single.zip` | train only; alpha 0.9, seed 3 | 4,631,365 bytes | Highest new single-model valid endpoint, 0.691442. One seed and adaptively selected from this screen; no all-data refit exists. |

Final-data logs record 12,479 fitting sequences; this is the existing loader's membership, so “train + valid” does not mean every row group in the raw files was necessarily consumed. The one-group omission has not been independently attributed; the review-status note preserves this limitation.

These are new candidates, not a demonstrated replacement for the historical TL3 public champion. The alpha .9 single and alpha .5 pair are effectively tied on valid. Fixed coefficients were not tuned. No final-data model was ranked using its fitted validation predictions.

## Evidence for selection

Official full valid: 1,873 sequences, 3,528,273 scored rows, `global_wp_clip2_v1`; all validation is repeatedly inspected development data. [Full scores](submission_checks/FULL_VALID_R1.json) match the independent reference metric implementation to <1e-10.

| Train-only endpoint | WP |
|---|---:|
| kd50 s3 | 0.690466215 |
| kd50 s4 | 0.690481734 |
| fixed 50/50 kd50 pair | 0.691422906 |
| kd90 s3 | 0.691442250 |

KD on top of since32 improves +0.003756/+0.002299 across the two fitted seeds (mean +0.003028). The since×KD interaction is +0.001587/−0.001786: the earlier synergy interpretation did not replicate. Two similar distilled endpoints do not prove removal of seed variation.

Excluded: row-dependent alpha (0.691117, one seed, below kd90 and no measured marginal blend value), six epochs (0.688870), no auxiliary heads (0.688568), second distillation round (0.689705), H384 (0.690212 and heavier serving), since8 (0.689727). The 50%/75% data experiments regress heavily over training, ending at 0.621511/0.662904; they are not competitive and do not isolate a general data-scaling law. Existing TL3C 0.692336 remains a stronger train-only ensemble on valid but already has a checked package/public receipt; no duplicate package is made.

## Technical checks and runtime

All three delivered hashes pass ZIP CRC, native ABI under Python 3.11, finite outputs and bitwise A-B-A sequence reset on the frozen 64-sequence panel. Every exported float32 NPZ tensor matches its fixed ep3 checkpoint exactly (only auxiliary output rows are removed). [Manifest](submission_checks/MANIFEST_R1.json) records checkpoint, provenance, runtime and ZIP hashes. [SHA256SUMS](submission_checks/SHA256SUMS) is also copied beside the ZIPs.

| Package | INT8 − Torch WP on panel | Relative RMS difference | Native callback, 64 sequences |
|---|---:|---:|---:|
| final pair | −0.0000116 | 0.439% | 63.20 s |
| final single s3 | +0.0000195 | 0.635% | 26.07 s |
| kd90 single s3 | −0.0000784 | 0.585% | 24.48 s |

[Parity report](submission_checks/NATIVE_PARITY_R1.json); [replay receipts](submission_checks/README.md). Full-valid Torch exports were computed, but native serving was replayed only on the 64-sequence panel. Final-data panel scores are technical diagnostics, not held-out quality. A full native competition-length timing was not rerun. Using the same-engine uploaded single's 15m27s and local timing ratios suggests roughly 30–40 min for the pair and around 15 min for a single; host runtime remains unmeasured for these hashes.

## Public receipts supplied by the user

| Portal ID | Artifact | WP | Duration | Memory |
|---|---|---:|---|---:|
| `1qn2bx0q` | train-only kd50 s3, SHA e5790b71… | 0.66108 | 15m27s | 628 MB |
| `1y4rw9o7` | teammate final fullmix | 0.66543 | 60m22s | 651 MB |
| `1h7kp3oq` | teammate w18 final | 0.66453 | 14m32s | 627 MB |
| `00kq8dw5` | tiedlead s5 INT8 | 0.66246 | 18m06s | 629 MB |

Source is the user's portal paste, not an independently queried portal. The fullmix did score despite the previously forecast timeout; its reported runtime gives no safety margin. Public outcomes are transfer evidence only. No local delta is added to a public score.

## Reproduction and review

- `grade_submissions.py` grades the fixed endpoints and performs parity assertions; run with root `PYTHONPATH=src` and bounded `scripts/limited` resources.
- Existing `research/campaign_2026-10-26/packaging_r1/package_tool.py` exported fixed ep3 members, built ZIPs and replayed the frozen panel; its hash is in the manifest.
- [Review status](SUBMISSION_AUDIT_R1.md): the independent reviewer returned no report within the collection window and was closed; no independent approval is claimed. Original logs and negative results remain in [RESULTS.md](RESULTS.md).
- No research training worker remains active after collection; no competition upload occurred.
