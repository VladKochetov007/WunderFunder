# F0 student-frontier H256×1 result review

**Date:** 2026-10-14  
**Scope:** independent receipt, comparison, and resource audit for the completed H256×1 seed-3 screen. This is not a fresh scorer run, numerical regrade, or training-seed replication. No Torch/NumPy import, model load, or provider action was used.

## Sources and identity checks

The terminal collection is [`h256l1_terminal_r6`](../../campaign_2026-10-14/cpu_student_frontier_r1/results/h256l1_terminal_r6/README.md); its `MANIFEST.json` SHA-256 is `8ee47a0110d8e362c5ec43859ae0bb72021f6de204a4d6cbb2e6425314a78101`. All 27 declared collected files matched their recorded byte counts and hashes, and each source file still matched its collected copy. The sequence was admitted against R6 plan SHA-256 `8b218d573b25f5ebe2ff88cade4d4141c4042e3c85f09abaa4bd6dfb94a22174`; the frozen consumer contract is SHA-256 `9476373fc3c32aea2ae48e9a2c48d211f0f152d13d086fa68efd1e8b1a02e768`.

I compared actual run provenance for H256×1 and the matched H256×2 control. Their serialized training arguments differ only by run name and recurrent layer count. Both use seed 3, fixed epoch 0, the same F0 split and fit-membership/order digests, 14,920 completed updates over 11,936 sequence exposures, zero AMP skips, and the same optimizer/schedule settings. The distillation cache metadata and verified train/valid target hashes match, including the same three clean-T3 epoch-2 source checkpoints; the propensity source and caches match; and both copy only the same H256×2 seed-3 epoch-5 normalization buffers, with identical mean/std hashes and no learned weights loaded. Thus the comparison isolates the intended one-versus-two recurrent-layer change under this recipe.

The grade receipts use `global_wp_clip2_v1`, the same fold-0 split hash, valid-target hash, row-group hash, and 808,863 scored rows. Candidate and reference checkpoints and full-valid FP32 prediction hashes bind to their respective fit receipts. The candidate's B16/C4000 profile validation and full-valid export completed; the grade identity validator passed. These are source/transport checks of the recorded grader result, not an independent recalculation of predictions or score.

## Result and interpretation

The reported WP is **0.6693463** for H256×1 and **0.6697776** for H256×2, a paired difference of **−0.0004313**. The reported 95% component-bootstrap interval is **[−0.0023043, +0.0013212]**, from 2,000 resamples over 141 components (468 groups). It crosses zero, so this panel does not resolve a quality difference; it neither rejects H256×1 nor establishes a win. The resampling summarizes uncertainty across evaluation components, not variation across fitting seeds. This is one seed on a repeatedly inspected F0 development panel, not a public forecast or clean final holdout. The historical 0.68314 endpoint used another recipe and is not part of this paired comparison; see the [separate control-grade review](CPU_STUDENT_FRONTIER_H256L2_F0_GRADE_REVIEW_20261014_R1.md).

The controller records one H256×1 fit elapsed time of 1,577.5 seconds versus 2,047.9 seconds for the H256×2 control. These are single-run fit-cost observations, not recurrent inference timings. Earlier random-weight Intel Q8 timing proxies do not measure these trained models and are outside this result review.

## Resource and handoff limits

The corrected terminal manifest now copies controller RAM fields as `peak_ram_bytes` and `peak_parent_ram_bytes`; the detailed attempts use these names. The fit peaked at 4.502 GB attempt-scoped RAM and 13.708 GB parent RAM; full-valid export peaked at 5.400 GB and 11.636 GB, respectively. These were within the admitted 5 GB fit and 6 GB export reservations. The fit and export controller receipts record zero OOM/max deltas, with 290,231 and 3,716 `memory.high` events, respectively. Full export recorded a 1.329 GB observed-VRAM peak under the 5 GB ceiling. The B16 profile's sampled controller VRAM is sparse; its 1.072 GB CUDA-reserved measurement is the more useful process-level figure, so the small sampled value is not treated as the true device peak.

Both R6 cache handoffs finished with unchanged CPU 500%, approximately 18/20 GB `MemoryHigh`/`MemoryMax`, and swap disabled. After-profile, the file counter fell from 10.729 GB to 8.138 GB against 2.590 GB requested; projected current plus the next 6 GB export reservation was 14.491 GB against the 17 GB committed-working limit. After-grade, the file counter fell from 10.308 GB to 8.244 GB against 2.064 GB requested; projected current plus the next 5 GB reservation was 13.461 GB against the same limit. These are measured reductions on this host, not a guarantee of future kernel reclaim amounts. R6 reused the already completed fit/profile stages; it did not train a second H256×1 model.

**Verdict:** retain H256×1 as a compact architecture-screen candidate if useful, with its lower one-run fit time labeled as a cost observation. The evidence does not make it a winner, a cross-seed result, or a serving-quality/runtime claim.
