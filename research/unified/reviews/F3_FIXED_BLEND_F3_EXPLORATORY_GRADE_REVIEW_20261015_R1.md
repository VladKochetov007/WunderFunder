---
kind: review
status: complete
created_utc: 2026-10-15T13:59:34Z
updated_utc: 2026-10-15T13:59:34Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-15/f3_teacher_export_r2/F3_FIXED_COMPARISON_R1.json
  - research/campaign_2026-10-15/f3_teacher_export_r2/F3_GRADE_INPUT_PREFLIGHT_R1.json
  - research/campaign_2026-10-15/f3_teacher_export_r2/SOURCE_BINDING_R2.json
  - artifacts/runs/clean_t3_f3_muonall_h384_e3_s3_r5/provenance.json
  - artifacts/runs/c20261003_combo_s3_f3/provenance.json
  - research/unified/inference/f3_qat_ancestry_r1/INVENTORY_R1.md
---

# F3 fixed teacher blend exploratory grade review

## Question or claim

Does the fixed 50/50 blend of the new F3 H384 teacher and the older F3 H256 combo show a reliable improvement on the frozen F3 evaluation panel, and are the two inputs cleanly bound to that panel?

## Mechanism and method

The grader selects the 505 F3 evaluation groups from the frozen fold-3 split and valid-target index, uses only valid-source component IDs, applies the official warmup/mask, and calls `global_wp_clip2_v1`. It imports `_member` from `score_fold0_panel.py`; although that module's docstring describes fold 0, the member validator itself accepts the supplied split, rejects direct valid-fit overlap, calls `split.assert_fit_groups` for component-level exclusion, and verifies checkpoint, split, row order, sidecar prediction hash, complete float32 shape, and finite values. The fixed blend is formed from raw member predictions before official clipping.

## Evidence and interpretation

The teacher is `clean_t3_f3_muonall_h384_e3_s3_r5@ep2`: its provenance and source binding identify fold 3, 10,539 train-fit and 1,368 valid-fit groups against 505 evaluation groups, seed 3, H384×2, fixed epoch 2 of 3, WMSE/chunk normalization, and the F3 propensity checkpoint `8898ef3d…f59b8`. Training used FP16 and completed 22,310 updates with 10 skips; the full-valid export is FP32. The older `c20261003_combo_s3_f3@ep3` provenance identifies the same split/fit counts and propensity checkpoint, but it is a different recipe: four epochs, H256×2, FP16 training, scratch initialization, and distillation from the F3 H128 seed-3 and seed-4 `since` teachers. Those teachers also record the F3 split and same train/valid partition. The older model has different normalization-buffer hashes and its prediction sidecar leaves inference precision unknown. This is an endpoint comparison, not an isolated teacher-family contrast.

On the 505-group, 145-component exposed panel (1,004,172 selected rows), the teacher scores 0.70801525 WP, the older combo 0.70954196, and the fixed half blend 0.71041684. The blend-minus-older estimate is +0.00087488, with a paired 2,000-resample component-bootstrap interval of [-0.00045517, +0.00201286] (seed 271828). The interval crosses zero and conditions on the fixed predictions; it does not include fitting variation. The equal weight was fixed after seeing the two single scores, so the interval also does not account for that post-result choice. The reported `p_le_0=0.0815` is the bootstrap fraction at or below zero, not a null-test p-value.

## Limits and validity

The split, metric, full-valid prediction identities, and teacher source/propensity ancestry are consistent with the reported grade. The older member's direct fit membership is disjoint from evaluation, and the prior F3 ancestry inventory records its two H128 distillation sources as using the same F3 split without held-out evaluation membership. The grade JSON binds both checkpoints and predictions but does not record either member's provenance hash; the source inspection here establishes the older run's present metadata, while a future grade should include its provenance/ancestor hashes for self-contained lineage traceability. This is exposed F3 development evidence with one seed per endpoint. It does not establish a general blend gain, teacher superiority, a distillation effect, or public transfer.

## Decision and next step

Retain the blend as an exploratory candidate for the already planned cross-fold and distinct-seed comparison. Do not call the +0.000875 point estimate a confirmed improvement. A stronger attribution requires matched teacher/student recipes and provenance-bound members on additional source-qualified folds and seeds; any further blend choice should be fixed before its confirming grade.

## Source records

- [Fixed F3 grade](../../campaign_2026-10-15/f3_teacher_export_r2/F3_FIXED_COMPARISON_R1.json), SHA-256 `65dc52a90b663e0523d96cc8b47c8e224c07dc9c275bbaed8995a5d7034d506d`.
- [Grade input preflight](../../campaign_2026-10-15/f3_teacher_export_r2/F3_GRADE_INPUT_PREFLIGHT_R1.json) and [teacher source binding](../../campaign_2026-10-15/f3_teacher_export_r2/SOURCE_BINDING_R2.json).
- Teacher provenance [c60719d1…c8da3](../../../artifacts/runs/clean_t3_f3_muonall_h384_e3_s3_r5/provenance.json) and older-combo provenance [c9cdaf39…42c19](../../../artifacts/runs/c20261003_combo_s3_f3/provenance.json); the combo records four distillation inputs from the two H128 F3 teachers.
- [Fold-3 split](../../campaign_2026-09-29/artifacts/splits/fold3.json), [official metric](../../../src/wnn/gmetric.py), [paired bootstrap](../../../src/wnn/evaluation.py), and [generic member validator](../../../scripts/score_fold0_panel.py).
- [Older H128 ancestry inventory](../inference/f3_qat_ancestry_r1/INVENTORY_R1.md); H128 seed-3 and seed-4 provenance hashes are `2ecb86cb…761e14` and `50db0a25…f052ca`.

This was a source/metadata audit; it did not load prediction or model payloads or independently rerun the grade.
