---
kind: review
status: complete
created_utc: 2026-10-24T13:39:24Z
updated_utc: 2026-10-24T13:39:24Z
hypothesis_ids: [H-ONLINE-RELATIONSHIPS-20261022-R1]
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/BATCH_MASS_RECIPE_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SOURCE_MANIFEST_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SOURCE_RECEIPT_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/SYNTHETIC_CHECK_R1.json
  - research/campaign_2026-10-24/resume4h_r1/feasibility_scout/batch_mass_audit_r1/audit_batch_mass.py
  - src/wnn/conditional_readout.py
  - src/wnn/training/online_relationship_fit.py
  - research/experiment_drafts/b300-fast-local-kaggle-20261023/kaggle/export_suffix_cache.py
---

# Independent source and measure review — batch-mass audit R1

## Disposition

**Bounded pass for the proposed descriptive audit.** I checked source and schema behavior, recipe/manifest identity, the synthetic receipt, and the header-only preparation receipt. I did not import or run the audit, hash or read the suffix payloads, or read targets. This supports the correctness of the specified measurement path, not any measured result.

## Identity and provenance

The frozen recipe SHA-256 is `999057ac546e050dca82c094894558be962f57fc3d4b5b28fc986d1cb5951ed3`; the source manifest SHA-256 is `1f5fc6e8ea36ca9a77745bae20b6e94f9e514004d22cc6584ebe482b0dc52a03`. The recipe binds the manifest and transport JSON hashes. I verified all 52 non-payload manifest entries against current bytes; the two suffix entries are intentionally excluded from this review and remain unverified at payload level.

The frozen synthetic checker matches the runner SHA `a006abee7639efab27e66cdd0ab96c497699eae73adb5f0ed6f71a70ebc880e4` and its own SHA is `d737eb55af8dddc30d8fb56c647056b5ddacc11fdc3c6b7c1f70471ee0a41f5a`. Its receipt covers clipping/raw-parent squared error, mass and distortion, pooled versus equal-batch normalization, zero and tiny positive masses, component shares, and a batch crossing a sequence boundary. This is a synthetic math check, not evidence about the private suffix values.

The source receipt explicitly reports that preparation read only each 128-byte NPY header and metadata. The folds are little-endian float32 C-order arrays with shapes `(1236, 20000, 5)` and `(1243, 20000, 5)`; payload hashes were sourced from the transport/export receipts and were not recomputed at preparation. The audit itself hashes each suffix while reading it and writes no result unless its full hash and byte length match the pinned transport record.

## Contract and calculations

The canonical fitter constructs fit references in selected train-group order followed by selected valid-group order. The cache writer stores one complete sequence per reference; its iterator preserves array order, and `_iter_row_batches` carries partial batches across sequence boundaries at 8,192 rows. The suffix exporter copies columns `[112:117]` per sequence without reordering. The audit checks ordered references, component closure, evaluation disjointness, transport/sidecar identities, and the five-column layout before scoring; `component_rows` follows the same concatenated sequence order across batches.

`primary_pair_wmse` clips the target to `[-2, 2]`; the fitter passes that same target as `weight_label`, so the effective weight is `abs(clipped_target) * row_weight`. It sums squared error and weight over both targets and clamps the batch denominator at `1e-8`. The audit matches that formula for the raw parent prediction without clipping predictions. Its equal-batch reconstruction averages each such batch loss, including exact zero-mass batches as zero; its pooled loss divides the summed numerators by total mass. The distortion factor is mean batch mass divided by that batch's mass and is undefined at zero mass.

The per-component pooled shares are component target-weight mass divided by total mass. The source-clamped shares sum each component's mass divided by `max(batch_mass, 1e-8)` and then normalize over all components. With positive ordinary-mass batches this gives batches equal total measure; exact zero-mass batches contribute zero, and sub-epsilon positive batches are attenuated by the source clamp. These are distributions of the declared loss weights, not per-component gradient attribution or model-quality measures.

All audit sums use float64 after reading float32 values. The recipe and output explicitly disclaim exact Torch FP32 reduction parity and optimizer-trajectory parity. The parent squared-error comparison is a fixed-parent diagnostic under two reductions, not a replay of the trained readout's losses or updates.

## Limits and next gate

No source/schema/math blocker was found. Both fits are in-sample development data; the differing fold and seed prevent interpreting their contrast as a seed-only or fold-only comparison. No evaluation labels, endpoint grade, clean validation effect, or optimizer behavior is established. Root admission is still required before reading the two suffix payloads; only a completed hash-verified run can support numerical batch-mass or parent-loss statements.
