---
kind: incident
status: complete
created_utc: 2026-10-15T17:55:39Z
updated_utc: 2026-10-15T18:39:10Z
hypothesis_ids: []
experiment_ids: []
evidence_paths:
  - research/campaign_2026-10-15/gru_parent_growth_r1/ROOT_CGFW_FIT_DELIVERY_AUTHORITY_R1.json
---

# User-requested research pause and bounded drain

## Question or claim

The latest user instruction is “pause research when its convenient”. It supersedes continuous research authorization and earlier conditional admissions. New research admission is closed immediately; healthy admitted work may reach a useful terminal checkpoint.

## Mechanism and method

The sole admitted drain was `warm-mechanism-fit-g-20261015-r1`, then its trusted CP1 state, full-valid FP32 export, and independent official grade against C. All are terminal. No F, W, S, initializer fit/profile, additional diagnostics or cloud retry started. The aggregate two-core, one-GPU, physical-scratch and memory limits were unchanged.

The global controller barrier `dfb6e4ddcce04d8ba4271a7fa0804d30` closed after the exact G export was admitted and remains present. It blocks new research until explicit user resume. The first grade invocation failed before producing output because the reference epoch was omitted; one corrected invocation explicitly used `@ep1` and completed under the unchanged scorer.

## Evidence and interpretation

C is complete and independently graded: official exposed-F0 WP 0.6812265132 versus original parent 0.6809412068; paired component interval crosses zero. G completed 14,920 attempted updates,14,914 completed and6 AMP skips; the trusted state gate returned `next_epoch=2`. Its export took121.56s and SHA256 is `4678e17ecb1ebfaf8e81ab54e55cbf5bd768c80c6fc92c339f5f9af514484a61` (299,680,128 bytes, finite FP32 `[1873,20000,2]`). Sidecar, checkpoint, split and row-group order all passed.

The official exposed-F0 score is G0.6812560749 versus C0.6812265132: paired delta+0.00002956165 with conditional95% component interval[−0.00038017998,+0.00037706961],2,000 resamples/141 components/468 groups/808,863 rows. This is one seed/two epochs on a repeatedly inspected panel, not quality promotion. The [independent review](../reviews/WARM_MECHANISM_G_CP1_GRADE_REVIEW_20261015_R1.md) has the comparison limits.

The Kaggle R5 version3 cold-shape profiles are terminal ERROR with zero optimizer updates. The strict cache-root validator rejects external symlink targets. No cloud job remains in that lane. Preserve the diagnosed successor proposal—stream-copy the two hash-bound arrays into their registered physical cache root, retain strict containment, test the actual cache consumer—but do not implement or dispatch it during pause.

The initializer actor sealed its code, static findings and runnable resume handoff without profile or fit. Source resolution binds modules to the pinned initializer tree. This remains static implementation evidence only.

## Limits and validity

Previously exposed folds remain exposed. No new architecture winner, teacher ceiling, public-score gain or CPU-policy speedup is established by this pause. CPU A/B/C/D ZIPs remain ready for user-owned uploads; no submission is uploaded by an agent.

## Decision and next step

### Restart-safety closeout, October15 18:39UTC

The bounded G drain, official grade and Git handoff are complete. The fit produced14,914 completed updates (six AMP skips); its trusted CP1 gate verified `next_epoch=2`. The full-valid FP32 export finished in121.56s with a finite `[1873,20000,2]` payload, SHA256 `4678e17ecb1ebfaf8e81ab54e55cbf5bd768c80c6fc92c339f5f9af514484a61`. Independent scoring found G0.6812560749 versus C0.6812265132, paired delta+0.00002956 and conditional component interval[−0.00038018,+0.00037707]; this exposed single-seed comparison does not promote G.

At18:39UTC no `wnn-*` or `run-*` service was active, `wnn-campaign.slice` had no child units, and `nvidia-smi` listed no compute process. Research/watchdog/cache-guard and campaign polling timers were stopped; the `STOP_ADMISSION` barrier `dfb6e4ddcce04d8ba4271a7fa0804d30` remains in place. The machine-readable fit, export and grade receipts, source review, ledgers, indexes and pause record were committed and pushed together in the closeout commit. Ignored model/data/prediction payloads remain on the laptop's physical disk and were not staged.

It is safe to restart the laptop now. Research remains paused; only an explicit user resume reopens admission.

Pause all research after the bounded drain; only explicit user resume reopens admission. Preserve all frozen sources, recipes, negative results and unrun arms. Update this record with the actual drain outcome rather than silently treating the estimate as completion.

## Source records

- [C/G/F/W delivery](../../campaign_2026-10-15/gru_parent_growth_r1/README.md), with G terminal and F/W held by the pause.
- [Student-quality evidence and S proposal](../../campaign_2026-10-15/student_quality_tradeoff_r1/README.md), including the independent G-versus-C grade.
- [CPU submission ZIP handoff](../../../submissions/10-14-cpu-policy-r1/README.md).
- [Cloud shape-pair packet](../../campaign_2026-10-15/student_shape_pair_f0_r1/README.md).
