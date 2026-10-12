# Seed-4 clean F0 control: final handoff

**Status:** the admitted six-epoch fit, fixed epoch-5 full-valid feature export, and official local F0 grade are complete. New admission is closed. Machine-readable evidence: [SEED4_CONTROL_FINAL_RECEIPT_R1.json](./SEED4_CONTROL_FINAL_RECEIPT_R1.json), SHA-256 430d1e5e574737b134515732ce785131ec13d65b82e2c769bc24d0ef7153ff98.

## Result and scope

Official global_wp_clip2_v1: **WP 0.681258421703229**, t0 0.674283747350372, t1 0.688233096056086; **468 exposed fold-0 groups**, **808,863 selected rows**. This is one seed-4 endpoint on a repeatedly exposed local panel, not a pristine holdout, cross-fold confirmation, blend screen, submission selection, or public-score evidence. Root owns scientific interpretation and append-only ledger updates.

## Recipe and ancestry

Run clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1 is student seed 4 with its own seed-matched initializer c20261003_combo_s4_f0@ep3 (checkpoint SHA 0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d). The frozen H-390 recipe uses GRU hidden 256 with 2 layers, AdamW, FP16, batch 16, chunk 2,000, unchanged six-epoch OneCycle, fixed epoch 5, and homogeneous T3 distillation alpha 0.9. No fit-time evaluation or schedule shortening was used.

Targets are own F0 H384 Muon-all epoch-2 outputs from seeds 3, 4, and 5, each coefficient 0.1. The producer applies the float32 mean scaled by 0.8125; student target is 0.1*y + 0.9*T. Fitting used 10,535 train.parquet groups plus 1,405 valid.parquet fit-valid groups. The 468 valid.parquet evaluation groups were excluded; same-source overlap is zero. Full-valid feature export covered 1,873 groups.

Six epochs made **44,760 optimizer attempts**, **44,740 completed updates**, and **20 AMP-skipped updates**. Per-epoch counters and actual P1/P2/P3 state bindings are in the JSON receipt and preserved history snapshots.

## Export and grade artifacts

Fixed epoch-5 checkpoint: /home/vlad/development/wunder/wnn_connectome_starterpack/artifacts/runs/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1/model_ep5.pt, SHA e6be82ff799f4ea2b62e770e9c340a8cfd8f98780d1c7dec4a76eab5a853d35c. Validated feature-only FP32 export: 1873×20,000×2, **299,680,128 bytes**, all finite, SHA f0689f9596d5319e7f5f2e54c5598168f17bcfd926e93e7724fc3878103edec8; metadata SHA d44e050d70a7aca2cda57bc895c5e8ded0178ef8fa7e9870ba1f5c06e40102a0. Grader aliases are hardlinks to those exact files, not prediction copies. [Official grade JSON](./f0_grade_r1/clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1.grade.json), SHA b9b189772581f35df1d3b3187b41d40e0410194161b266cba22033b90781c3fc. Evaluation subset: /home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/f0_grade_r1/eval468_pred_clean_t3_f0_control_s4_a9_t3r5_r3_p3_envr1_ep5.npy, SHA 6703664da6ba27e78134b70795a4890ef147df958fdd62173d70397b2d1d11c7.

## Execution and resources

P1/P2/P3 elapsed 2023.5/1998.1/2116.9s. Worker RAM peaks: 4,952,309,760/4,952,342,528/4,952,354,816 bytes; observed NVML VRAM peaks: 2,638,217,216/2,638,217,216/2,638,217,216 bytes. Export finished exit 0 in 106.153s; worker RAM 4,950,261,760 bytes and observed NVML VRAM 1,323,302,912 bytes. Fit/export scopes used one CPU, 5.5 GB RAM max, 4.3 GB allocator ceiling, 5 GB observed VRAM ceiling, zero loaders and swap disabled.

Worker and parent-local OOM/OOM-kill/max counters were zero. memory.events high/max are event counts, not bytes or pages. P1 had parent-hierarchical max=7 events while worker and parent-local max=0; OOM/kill remained zero. P2/P3/export parent-hierarchical max=0. Grade ran under a 1-CPU/1.5-GB/120-second/no-swap scope and exited 0 in 3.217 seconds. The short-lived scope did not retain a post-exit MemoryPeak, so no grade-process peak is claimed. The export-validator R2 receipt also carries a contradictory ru_maxrss report above its configured 500 MB support cap; unchanged parent-local events make that field an untrusted measurement, not a claimed process peak.

## Consumer correction and label access

The first export-validator receipt is preserved: R1 failed with KeyError run_name because the frozen contract does not contain that optional key. R2 derives the expected run from the pinned worker argv and passes all eight checks; no producer, source, contract, or training science changed. See [R1 receipt](./CP5_FULL_VALID_EXPORT_TERMINAL_VALIDATION_R1.json), [R2 receipt](./CP5_FULL_VALID_EXPORT_TERMINAL_VALIDATION_R2.json), and both validator scripts.

The feature-only exporter did not read labels. R2 accessed only row_group from valid_targets.npz. A prior pre-grade shape probe accessed/decompressed the NPZ y member only to print shape; no values were inspected or used and no score was computed. The final official grade intentionally read y and mask for the 468 evaluation groups.

## Linked evidence

- Root release: [ROOT_SEED4_CONDITIONAL_RELEASE_R1.json](</home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1/ROOT_SEED4_CONDITIONAL_RELEASE_R1.json>) (SHA 3c4fd04c82e0386ce8071238a678009bf0350594460d83c5fc87c00d1dc84b78).
- [Frozen fit contract](./FIT_CONTRACT_R1.json) and [launch manifest](./LAUNCH_MANIFEST_R1.json).
- Stage terminal receipts: [P1](./P1_TERMINAL_VALIDATION_R3.json), [P2](./P2_TERMINAL_VALIDATION_R1.json), [P3](./P3_TERMINAL_VALIDATION_R1.json).
- [CP5 export contract](./CP5_EXPORT_BOUND_STAGE_R1.json).
- [P1/P2/P3 evidence preservation manifest](./preserved_stage_evidence_r1/PRESERVED_STAGE_EVIDENCE_MANIFEST_R1.json); all 12 history, provenance, attempt, and log copies match source hashes.
- Prediction, metadata, checkpoint, state and grade-subset paths with hashes are in the JSON receipt. No agent-created files were left only in /tmp.
