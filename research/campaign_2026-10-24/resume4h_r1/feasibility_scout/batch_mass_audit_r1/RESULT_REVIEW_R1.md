---
kind: review
status: complete
created_utc: 2026-10-24T13:50:56Z
updated_utc: 2026-10-24T13:54:35Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: [E-ONLINE-RELATIONSHIPS-20261022-R1]
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RESULT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/EXECUTION_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RECIPE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SOURCE_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SOURCE_MANIFEST_R1.json
---

# Independent result review — batch-mass audit R1

## Disposition

The R1 descriptive output passes the bounded JSON and identity consistency checks. The two suffix digests recorded by the completed run exactly match the expected hashes in the transport-backed source receipt and frozen fold recipe. I did not read or rehash the suffix payloads; this review independently cross-checked the saved identities and aggregate arithmetic only. The runner's `source_suffix_sha256_verified` fields record that it streamed and hashed both complete payloads before accepting the results.

This measures the original fitting objective on its **in-sample fit data**. It does not grade an endpoint, evaluate generalization, compare optimizer behavior, or establish that the source objective harmed quality.

## Identity, completion, and scope

Current file hashes: result JSON `a26a92e60ff97a710b94d85a7c1402dc3210948aab659a4de98347ef77684e8d`; recipe `999057ac546e050dca82c094894558be962f57fc3d4b5b28fc986d1cb5951ed3`; source manifest `1f5fc6e8ea36ca9a77745bae20b6e94f9e514004d22cc6584ebe482b0dc52a03`; execution receipt `61721f2fb67c06c64bc9e35d442406b3b85b35a33290bbca142da51dcd9869b8`.

The F0 digest `a273d2fa5fe30f86c209e2899cceca636480954d7f69daf79fe402b6aaaedb81` matches the pinned `(1236, 20000, 5)` little-endian float32 suffix (494,400,128 bytes including the NPY header). F1 digest `2dfbd54cb7b26fb628470afe4a013b60302c2ab3d9b4b2d4eb533c100b35ad63` matches `(1243, 20000, 5)` (497,200,128 bytes). The fold membership digests match the recipe; row counts, 8,192-row batches, final partial batches, component counts, and totals are consistent. Component-share sums, HHI, effective component counts, largest shares, and top-decile shares recompute from the saved per-component JSON records.

The bounded run returned code 0 in 6.942 seconds at the worker-wrapper level; its result timer records 6.827 seconds, a 0.115-second difference between timing boundaries. Peak memory was 399,998,976 bytes under the 400,000,000-byte cap, with 1,354 memory-max events, zero OOM events, and zero swap. The result states zero fits, no Torch/model imports, and no evaluation-label reads. The worker's float64 reconstruction from pinned FP32 suffix values is not Torch FP32 reduction parity.

## Aggregate checks

| Measure | F0 (seed 1001) | F1 (seed 1002) |
|---|---:|---:|
| Fit rows / batches | 24,720,000 / 3,018 | 24,860,000 / 3,035 |
| Positive / zero-mass batches | 3,018 / 0 | 3,035 / 0 |
| Mean joint batch mass | 1,241.025 | 1,039.737 |
| `mean batch mass / batch mass` factor range | 0.109–9.743 | 0.104–8.291 |
| Pooled target-mass component HHI / effective count | 0.008408 / 118.94 | 0.007974 / 125.41 |
| Equal-batch component HHI / effective count | 0.005692 / 175.69 | 0.005646 / 177.11 |
| Pooled / equal-batch top 10% component mass | 38.94% / 30.33% | 37.57% / 30.53% |
| Pooled parent loss | 1.211032 | 1.298912 |
| Equal-batch normalized parent loss | 1.058217 | 1.100586 |
| Equal-batch relative to pooled | −12.62% | −15.27% |

The component aggregates independently reconstruct from the 315 F0 and 314 F1 component records. The two target masses sum to joint mass in both folds, and the target-specific contributions sum to each reported joint loss. Every batch has positive mass (minimum 127.37 F0, 125.41 F1), far above the `1e-8` denominator clamp, so zero-mass omission and clamp attenuation do not affect these observed batches.

## Interpretation and limits

The distortion factor is the relative per-row weight multiplier between the equal-batch normalized and pooled fixed-parameter loss measures: for batch `b`, it is `(total mass / batch count) / batch mass[b]`. Thus the observed range shows how much these two averaged losses reweight each row by its batch. Differentiating both losses at the same fixed parameters preserves that relative factor. It does not establish the resulting AdamW updates or trajectory as parameters evolve batch by batch. Under the pooled mass measure, the component distribution is more concentrated than under the source's equal-batch normalized measure; this describes two measures over the fixed fit suffixes, not causal component influence or a benefit from changing the loss.

The lower equal-batch parent squared-error aggregate is descriptive. It does not imply that equal-batch normalization is harmful, that pooled normalization would improve or reduce official `global_wp_clip2_v1`, or that either alternative would change trained weights in a predictable way. No training was repeated and no evaluation endpoint was scored. Fold and seed change together, so these two summaries do not isolate either axis. Existing [source/math review](CODE_REVIEW_R1.md) confirms the audited formula against the fitter, row ordering, transport columns, and component-measure interpretation; it explicitly does not claim optimizer parity.
