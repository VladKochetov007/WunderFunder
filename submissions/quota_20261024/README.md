# Submission quota inventory — 2026-10-24

This folder holds the pre-build inventory and two fixed-weight exploratory TL3
packages prepared in response to the request for plural quota candidates.
[INVENTORY_R1.json](INVENTORY_R1.json) is the source-stage record: it identifies
the existing optional canary, pins the H390 source members and their exposure,
and records saved diagnostics and evidence limits. It records accurately that
no ZIP had been built at inventory time.

## Candidate status

The strongest existing unsubmitted package located is
[H390 S3 + H388 teacher](../10-11-67/11_h390_student_s3_h388_teacher_s5_all_q8vnni.zip)
(12,914,019 bytes, SHA-256
`d83723831277cf7327ab84ba781a2f2ebe883aca40cd1f7bc6a222e30be28284`). Its
saved exact-ZIP Python 3.11 and 3.13 callback panels pass finite-output and
reset/replay checks. It has no measured held-out score: the nearby 0.672574
fold1 blend is a different endpoint. Existing readiness records mark it
optional and do not record an upload; this check did not query the live portal.

Two new, distinct endpoints are structurally feasible from the exact
[original TL3 ZIP](../team_tl3_original_20261018.zip):

1. Replace its 0.34 S5 member with H390 S3, retaining TL5/TL6 at 0.33 each.
   Estimated ZIP size from current entries: 12,812,144 bytes.
2. Replace the 0.34 S5 member with H390 S3 and S4 at 0.17 each, retaining
   TL5/TL6 at 0.33 each. Estimated ZIP size: 16,720,424 bytes.

Both H390 NPZs have the exact key set and tensor shapes/dtypes expected by the
old TL3 wrapper: two-layer H256 GRU, 211 inputs, two outputs, FP32 arrays. Their
16 array payloads are all distinct from TL3 S5; S3 and S4 are distinct from
each other; their mean/std arrays differ from TL3 S5. This supports structural
compatibility only. The original TL3 package calls `wnn_cpp` and dynamically
quantizes its non-FP32 members; the separate H390 package uses `wnn_native`.
No parity, quantization quality, or callback runtime has been measured for the
H390 members inside the TL3 wrapper. Root/packager owns exact-ZIP validation.

S3/S4 saved diagnostics cover 1,873 validation row groups each in FP32, but the
members were trained on train+valid. These predictions are exposed diagnostics,
not clean held-out evidence or an exact score for either proposed package. No
expected public-score gain is supported. Older public results for other H390
compositions are listed in the receipt and must not be treated as scores for
these endpoints.

## Built packages

The byte-packaging recipe was frozen before ZIP creation in
[BUILD_RECIPE_R1.json](BUILD_RECIPE_R1.json) (SHA-256
`6838151cad4dea2127a84e00dfe17c6932860164c03d94df7074c7b79ca22954`). The
stdlib-only [builder](build_packages.py) verifies pinned source entries and
creates outputs exclusively. Its SHA-256 is
`c1a974a49b051270de9a5e4389e939e769b34a9c86caa1360353e09380c0dd32`.

| Package | Fixed members | Bytes | SHA-256 |
|---|---|---:|---|
| [TL3 + H390 S3](tl3_h390_s3_20261024.zip) | H390 S3 .34, TL5 .33, TL6 .33 | 12,537,972 | `56ebf18d26187541e539a31078a50dcbeb698e395ca0ecde5e1174aa2e929ded` |
| [TL3 + H390 S3/S4](tl3_h390_s3_s4_20261024.zip) | H390 S3 .17, H390 S4 .17, TL5 .33, TL6 .33 | 16,161,940 | `8bf614168d97d50b58557db3a1f5befa16f4912b5b8b1fbf609b087f62f96be2` |

The [build result record](BUILD_RESULTS_R1.json) and per-package receipts
record entry hashes, CRC, fixed member maps, source hashes, and the 20 MB size
check. The original `solution.py`, all three native libraries, and the TL5/TL6
weight bytes are unchanged in both ZIPs. The first package changes only the S5
member bytes and member map; the second also adds the existing S4 NPZ.

These are prepared packages, not validated releases. TL3's `wnn_cpp` ABI,
reset/replay, relative runtime, and exact composite grade remain unchecked.
H390 was fit on train+valid, so its validation exposure is known; TL5/TL6
exposure is not established by the preserved source ZIP. No expected
public-score gain is supported, and neither package has been uploaded.

## Sources

- [Quota submission authority](../../research/campaign_2026-10-24/resume4h_r1/QUOTA_SUBMISSION_AUTHORITY_R1.json) records the user request and fixed endpoints.
- [October 22 public receipt](https://github.com/VladKochetov007/WunderFunder/blob/c3e208b387ead0ec967f9abc4b14275eb8ba94c5/research/campaign_2026-10-22/quota_retained_pair_r1/PUBLIC_RECEIPT_R1.json) contains the user-supplied TL3 reference score and related submissions; it was not live-reverified here.
- [H390/H388 exact package gates](../10-11-exploit-now_r2/h390_student_h388_teacher_pair_q8/PACKAGE_GATE_RECONCILIATION_R2.json) report package ABI/callback checks and explicitly leave the exact candidate's held-out score null.
- [Earlier H390 public results](../10-12-best-r1/PUBLIC_RESULTS_20261012_R1.json) identify different already-uploaded packages to exclude from byte-repeat recommendations.

## Completed native check and handoff

The [exact callback results and handoff](../../research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/RESULTS_AND_HANDOFF_R1.md) deliver the S3/S4 ZIP first, then S3, under the explicit exploratory quota request. All three exact packages pass ABI, finite-output, bitwise-reset and identical-input checks across64sequences. Pair deltas versus TL3 are +0.003742/F0 and +0.000205/F2; callback time is about24% greater on this host. The [handoff receipt](../../research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/HANDOFF_RECEIPT_R1.json) binds actual ZIP identities, grade and independent review. These are fitted/unknown-ancestry sampled diagnostics, not clean CV or a public forecast. Neither ZIP has been uploaded by this task.
