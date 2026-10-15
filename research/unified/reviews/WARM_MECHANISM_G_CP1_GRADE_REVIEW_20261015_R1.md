---
kind: review
status: complete
created_utc: 2026-10-15T18:36:00Z
updated_utc: 2026-10-15T18:36:00Z
hypothesis_ids: [H-STUDENT-PARENT-GROWTH-20261015, H-087]
experiment_ids: [E-WARM-MECHANISM-G-CP1-20261015-R1]
evidence_paths:
  - research/campaign_2026-10-15/student_quality_tradeoff_r1/WARM_MECHANISM_F0_S3_G_R3_CP1_PAIRED_GRADE_R1.json
  - research/campaign_2026-10-15/gru_parent_growth_r1/G_EXPORT_TERMINAL_VALIDATION_R1.json
---

# Independent grade review: group-age student G at CP1

## Claim reviewed

Does adding the causal group-age input improve the original H256×2 student after the frozen two-epoch continuation, compared with the matched C arm without research features?

## Checks and result

I independently invoked the unchanged `scripts/score_fold0_panel.py` on the exact exported G and C CP1 predictions. Its SHA256 is `0f67ac06cd769a582901b854737528c89e9a7738c3057acf68d7f99eddbfa2d0`. The scorer bindings match both checkpoint and prediction hashes, the same split manifest, and the same 1,873-row-group order. G's feature-only FP32 export is finite and has shape `[1873,20000,2]`; its prediction payload is299,680,128bytes and its sidecar hash matches the payload.

On the official `global_wp_clip2_v1` metric, G scores0.6812560749 and C scores0.6812265132 over468 exposed F0 evaluation groups,141 components and808,863 scored rows. Paired component-bootstrap delta is+0.00002956165 with95% interval[−0.0003801800,+0.0003770696] (2,000 resamples; seed271828). The result shows no resolved quality advantage for G at this checkpoint.

The first scorer invocation used a bare reference run name and failed during argument parsing before writing any result. The corrected command pinned `warm_mechanism_f0_s3_c_r3@ep1`; it completed successfully. No scorer implementation or grade data were changed.

## Limits and disposition

This is one seed, two epochs, and a repeatedly inspected F0 panel. The two variants share the frozen parent, clean-T3 targets, fitting membership and training recipe; this is a useful matched input-feature comparison, not a multi-seed recipe confirmation or fresh holdout. Retain the group-age arm as a tested candidate with near-equal observed quality; do not promote it on this result alone or infer that causal age features are ineffective generally.

The user-requested pause remains in force. The exact C/G/F/W wave stops after G; F, W, S and initializer fits/profiles remain unrun. No additional experiment is recommended or admitted here.

## Source records

- [Machine-readable paired grade](../../campaign_2026-10-15/student_quality_tradeoff_r1/WARM_MECHANISM_F0_S3_G_R3_CP1_PAIRED_GRADE_R1.json).
- [Export validation](../../campaign_2026-10-15/gru_parent_growth_r1/G_EXPORT_TERMINAL_VALIDATION_R1.json).
- [Fit state gate](../../campaign_2026-10-15/gru_parent_growth_r1/G_FIT_TERMINAL_STATE_GATE_R1.json).
- [Pause closeout and restart check](../operations/RESEARCH_PAUSE_20261015_R1.md).
