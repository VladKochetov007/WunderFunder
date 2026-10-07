# Public C1 transfer: independent interpretation

Reviewed 2026-10-07 from the public-result receipt, E511 grader/recipe, E524 delivery evidence, the incumbent package provenance, and H230's current execution contract/receipts. This memo adds interpretation only; it does not replace the portal receipt or change any fit, score, allocation, or ledger.

## What is measured

The user-reported portal rows in `E524_PUBLIC_RESULTS_20261007_V1.json` show both exact E524 single ZIP submissions below the existing 0.6660 incumbent:

| Submitted ZIP | Portal ID | Public score | Delta to 0.6660 | Portal runtime / memory |
|---|---|---:|---:|---|
| `c20261007_c1_s3_i16.zip` | `1n6uxwb6` | 0.6647 | -0.0013 | 33m11s / 624 MB |
| `c20261007_c1_s4_i16.zip` | `1cymqpkk` | 0.6654 | -0.0006 | 33m07s / 624 MB |

The result receipt binds the user-reported portal entries to local ZIP names and digests; the portal did not supply ZIP digests. Neither upload improves the public best. Do not reupload these identical files. The independently verified own best remains 0.6660; Maxim's reported 0.6674 remains unverified here.

E511 answers a different question. Its fixed grader compared each C1 checkpoint with its same-seed C0 checkpoint and compared the equal-weight seed pair, on 468 previously exposed fold-0 evaluation sequences (141 components; 808,863 scored rows) under `global_wp_clip2_v1`. C1's single-seed local deltas were +0.001666 and +0.001676; the pair delta was +0.001023 with a 95% component-bootstrap interval of [-0.001473, +0.004619]. This is a positive but imprecise development comparison, not a C1-versus-public-incumbent comparison and not a public-transfer forecast.

## Why the local and public signs can differ

The E524 C1 singles are one-member, i16 Rust packages containing `bd26_c1v2_s3_f0@ep1` or `bd26_c1v2_s4_f0@ep1`. The frozen E511 recipe trains two-layer GRU-256 students for two epochs, selects fixed epoch 1, initializes each from its same-seed own `c20261003_combo` epoch 3, and fits on 10,535 train plus 1,405 valid-fit groups from fold 0. The 468 fold-0 evaluation groups are excluded from those fits. Its main target is `0.3 * clipped label + 0.7 * clipped mean teacher`; the two strong-teacher predictions have effective coefficient 0.40625 each after the legacy teacher-list averaging. E511 C0 is the matched same-seed comparator for this treatment.

The prior public-best package is a different trained endpoint: its single packaged weight file is sourced from `c20261004_combo_s4_final@ep3|c20261004_combo_head_s4_final@ep0`. The base run provenance declares a conditional rank-8 adapter, four epochs, Muon, batch 32/chunk 1000, `distill_alpha=0.5` from `ab67_final_since_s3,s4`, and `prop_v1`; its recorded membership is 10,606 train plus 1,505 valid-fit groups with a separate 368-group evaluation set and no fold-0 split-manifest binding. E511 does not grade this incumbent against C1 on its exposed fold. So the public regression is real for the submitted C1 ZIPs, but the local matched C1-vs-C0 gain does not identify why C1 is below this materially different incumbent.

There is some evidence against a gross packaging/runtime regression: E524's independent provenance review found its `solution.py` and native libraries byte-identical to the incumbent package, and each new ZIP contains the exact epoch-1 model export as a single i16 member. The E524 CPU-float32 parity check passed its 0.0003 max-error gate (C1 seed 3: 0.0000753; seed 4: 0.0001038). On a fixed 20-sequence diagnostic, native i16 versus the cached GPU-FP16 metric differed by about +0.000216 for seed 3 and +0.000010 for seed 4. That sample is too small to rule out an endpoint/precision contribution on the public set, but it gives no evidence that packaging alone explains the observed shortfall.

Generalization shift, training recipe/teacher differences, fit-membership differences, and evaluation-population differences remain possible explanations, not findings. The portal scalar scores contain no per-sequence evidence to separate them. In particular, do not add E511's exposed-fold delta to a public score or infer that the heterogeneous-distillation family failed.

## H230 readiness and next actions

H230 is the fuller own-weight heterogeneous student: an eight-source/four-family teacher cache, raw family targets, the same 10,535 + 1,405 fold-0 fitting groups and 468 excluded evaluation groups, and a GRU-256x2 student with 31 outputs. Its main target remains the C1-style 0.3/0.7 blend; the family head is an additional 0.1 loss. The verified cache, source bundle and representative full-shape profile already exist. The seed-3 student fit, fixed epoch-1 checkpoint, prediction export and grading remain unrun; its reviewed 9-GB/3-CPU/2,400-second fit successor is prepared but still requires fresh root admission and live resource/lock checks.

The public C1 result does not add a prerequisite to H230. Keep the already reviewed source/cache/profile evidence; do not rebuild those stages or wait for a new C1 explanation. The existing live memory/VRAM/quota/ledger/source checks and root-owned seed-3 admission are the necessary gates. Once admitted, the two highest-value follow-ups are:

1. **Run the original H230 seed-3 candidate through its frozen epoch-1 fit, valid export, and independent official grader.** Grade only the 468 excluded fold-0 groups, with the existing C1/C0 results and incumbent included on that same mask where artifacts permit. Preserve this as exposed development evidence, not a public forecast. A fresh attempt and resource admission are still required; no source/cache redo or additional public-result review should gate it.
2. **Run the already prepared matched family-loss-off H230 control after seed 3 is terminal.** Keep the same cache, seed, initialization, membership, optimizer, schedule, main target and all 31 outputs; change only the family-loss coefficient to zero, as specified in `H230_MATCHED_FAMILY_OFF_CONTROL_V1.json`. Pair-grade on the same evaluation groups. This isolates whether raw per-family targets add value beyond the H230 main-distillation treatment. Keep it serial and do not make it a prerequisite to admitting the original H230 candidate.

Retain the online-learning family as a separate exploratory lineage. The C1 portal rows neither test nor reject its causal dynamics/fast-weight hypotheses; no online-learning task fit or public forecast is established.

## Evidence links

- [Portal-result receipt](E524_PUBLIC_RESULTS_20261007_V1.json)
- [E511 matched C1-vs-C0 report](../local/E511_C1V2_C0_EVALUATION_v1/REPORT.md) and [frozen C1 recipe](../local/E511_C1_MINIMAL_PAIRED.json)
- [E524 package README](README.md), [delivered-ZIP validation](E524_DELIVERED_ZIP_VALIDATION_RECEIPT_V1.json), and [independent provenance review](C1_E524_PROVENANCE_EVIDENCE_REVIEW_20261007.md)
- [H230 execution contract V2](../wide_baseline_heterogeneous_H230/H230_EXECUTION_CONTRACT_V2.md), [profile result](../wide_baseline_heterogeneous_H230/H230_PROFILE_RESULT_V1.json), [fit resource successor](../wide_baseline_heterogeneous_H230/H230_SEED3_FIT_RESOURCE_SUCCESSOR_V2.json), and [matched family-off control](../wide_baseline_heterogeneous_H230/H230_MATCHED_FAMILY_OFF_CONTROL_V1.json)
