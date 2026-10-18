# Exploratory F0 submission, 18 October 2026

The user requested an optimal submission from the completed F0 endpoints. This
is an explicit release exception for this exposed fold, while the rank1 CP30
comparison and healthy F1 training continue. No competition upload is requested.

## Selection

The [frozen plan](PLAN.json) specifies five scalar mixtures of the independently
verified epoch5 wide H512×1 and deep H384×2 checkpoints. The same coefficient
applies to both targets. On all 468 F0 groups / 141 components / 808,863 scored
rows, the best tested mixture is **75% wide + 25% deep**, WP **0.6738509442**.

| Wide / deep | Official FP32 WP |
|---|---:|
| 0 / 100% | 0.6721087673 |
| 25 / 75% | 0.6731209255 |
| 50 / 50% | 0.6737010465 |
| 75 / 25% | **0.6738509442** |
| 100 / 0% | 0.6735757510 |

This is selection on a previously exposed F0 cell, not independent validation,
CP30, proof of an architecture win, or a public score forecast. “Best” means
best of this fixed five-point comparison, not a global optimum over all recipes.

The [blend grade](receipts/BLEND_SELECTION.json) pins prediction arrays,
checkpoints, provenance, split, targets and source receipts. Both scalar endpoint
scores reproduce the prior independent grade. An independent direct float64
weighted-Pearson rescore agrees within 4.5e-16. See the
[independent selection and tensor review](receipts/INDEPENDENT_REVIEW.json).

## Actual package checks

The [FP32 mixture build](receipts/MIX75_F32_BUILD.json) is 17,480,572 bytes.
Its [first two complete sequences](receipts/F32_PANEL.json) pass Linux
Python3.11 callbacks and exact A-B-A reset. The maximum prediction-required
difference versus canonical GPU predictions is 6.03e-6; native fast activations
are approximate. Its measured CPU runtime is too slow on Beleriand.

The [Q8 mixture build](receipts/MIX75_Q8_BUILD.json) uses the same checkpoints,
coefficients and fixed member list with dynamic integer inference at load.
There is no retraining, calibration fit, output-dependent routing or wall-clock
governor. The [Q8 panel](receipts/Q8_PANEL.json) passes the same callback/reset
checks and is about four times faster than FP32. Its full F0 replay is pending.

The [same-host incumbent panel](receipts/INCUMBENT_PANEL.json) checks the exact
locally archived `xteam_h4.zip`. The new Q8 pair is 1.072× its two-sequence time.
Applied to the user-transcribed historical 3016-second incumbent runtime, this
gives a **53.9-minute inference estimate**, not a competition-server guarantee.
Neither the historical portal nor the uploaded payload hash was independently
verified. Absolute Beleriand extrapolation is 137 minutes on its Xeon Gold6230;
the host is not the competition scorer. Both numbers are retained rather than
treating the favorable calibration as proof.

Full source-qualified F0 replay is active under
`computer-vision-homework-f0-submit-release-replay-v1.service`. It first runs
static ELF and v2/AVX2 callback checks, then all 9.36 million F0 rows through the
actual Q8 ZIP. The [fixed launcher](code/replay_release.sh) started under1CPU; its quota was reduced to0.75CPU after the [F1 admission incident](../priority25_execution_r4/f1_recovery_r1/README.md). It retains
2GB RAM, swap0 and a 5400-second deadline inside the existing two-core campaign
slice. No extra GPU worker is created. Its elapsed time spans both quotas and must not be used as a one-core benchmark. Final grade and delivery are pending.

## Evidence and reproduction

- [Code](code/README.md): create-only extraction, five-way selector, reference
  construction and actual ZIP callback replay; no trainer changes.
- [Receipts](receipts/README.md): archive identities, selection, package builds,
  executed checks, support bounds and independent review.
- [Local asset manifest](receipts/LOCAL_ASSETS.json): exact terminal checkpoint,
  provenance, native-library and wrapper hashes. The canonical packager exported
  these weights; independent CPU reload matched all 12 wide and 16 deep tensors.
- [Extraction](receipts/EXTRACTION.json): direct original-parquet scored labels
  and hash-pinned prediction arrays, plus eight complete diagnostic sequences.
- [Reference construction](receipts/REFERENCE.json): full fixed 75/25 FP32
  reference, using the same multiply-then-add order as selection.
- [Support execution](receipts/SUPPORT_EXECUTION_V1.json): serial successful
  support units and the active replay allocation. Own exited units were captured
  and stopped; training and foreign services were not modified.
- [Wide-only engineering build](receipts/WIDE_F32_BUILD.json): retained initial
  package, not the selected release.

Large assets remain in `/Users/nb/Documents/wunder/runtime-20261018/f0-submission-r1`
and `/home/grinchenkoeg/computer-vision-homework-20261018/submission-f0-r1`.
The original immutable F0 archives remain under `checkpoint_snapshots` and
`result_snapshots/rr18_r01_f0_s1001_e6` on Beleriand.

Erratum: the extraction receipt's `parquet_full_hash` text mentions checking the
footer. This extractor checked file size, row-group count and selected sequence
identity/order, and inherited the full source hash and sidecar ordering check
from the archived grade. It did not independently hash the full parquet or its
footer. Full native replay separately computes the original parquet SHA.
