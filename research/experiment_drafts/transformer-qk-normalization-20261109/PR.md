# [Experiment] Affine-free QK RMS normalization on B300

1. **Status, lineage and owner.** Published [draft PR #75](https://github.com/VladKochetov007/WunderFunder/pull/75) prepares four fresh fits,
   owned by ROOT, targeting `research-team`. Parent is PR72 head
   `4021de8ec5130143c207319c73a102a453eb911e`; target snapshot is
   `7e4074314a62d0907c9608f9220cf72958708a2d`. This is stacked on PR72, so the
   full PR includes its runtime. Review the incremental QK diff separately.
   Hypothesis H-B300-TRANSFORMER-QK-RMS-20261109; experiment
   E-B300-TRANSFORMER-QK-RMS-20261109. All new fits/profiles are UNRUN; no local
   tests, model imports, training or inference and no live-job changes.

2. **Why it might work.** Normalize newly projected Q/K head vectors in FP32
   using `x / sqrt(mean(x**2) + 1e-6)`, cast back, then apply RoPE. The cache
   stores normalized rotated K once; values are untouched. A fixed norm may
   make attention less sensitive to QKV magnitude changes. But magnitude may
   carry useful information, and the treatment changes temperature as well:
   initial logit standard deviation may be roughly3x the control under default
   Linear initialization. We have not measured collapse, logits or entropy.
   This is a package-quality comparison, not pure causal identification of
   saturation prevention. No parameters/buffers/RNG draws are added.

   [Henry et al.](https://arxiv.org/abs/2010.04245) use L2-normalized Q/K with a
   learned scale; [ViT-22B](https://arxiv.org/abs/2302.05442) uses Q/K LayerNorm
   in large vision models. Our affine-free fixed-RMS variant reproduces neither
   and their results do not establish financial forecasting gains. PR64 recency,
   PR65 output gates, PR66 SwiGLU and PR74 input convolution are distinct.
   GALAM QK-Norm is an antecedent, not a matched result for this Transformer.

3. **Compute advantage and cost.** Four new teacher fits, two per arm; zero
   student fits and zero reused controls. Each fit is4E/3280 updates, total16
   model-epochs/13,120 scientific updates. Eight50-update engineering profiles
   add400 diagnostic updates, not scientific replications. Two B300s per fit;
   two paired seed streams can occupy separate verified GPU pairs. Actual
   availability, wall time, memory and disk demand are not assumed. No new
   resource allocation, queue manager or automatic follow-up is implemented.

4. **Frozen recipe.** Exact PR72 C0 except seed and optional `model.qk_norm`.
   Transformer768x24,12heads,FF4,context1024,dropout0,hint0; per-rank B128,
   world2/global256,chunk256,gradient horizon256,accumulation4. Muon LR3e-4,
   momentum.95,NS5,WD.01; AdamW betas.9/.95; clip1; warmup5%=164 updates,
   cosine floor.1. BF16/BF16x9 with FP32 masters/loss/optimizer state.
   Cache-free standalone_labels_wmse_v2, all_fit normalization,
   all_after_warmup mask. Configured final E4 is primary; all epochs are
   retained diagnostically. No outer-valid peak selection or quantization.

5. **Bounded panel.** `plain_s203301`, `rms_s203301`, `plain_s203302`,
   `rms_s203302`. Same physical GPU pair/order and software within each seed;
   same source/data/parameter count/optimizer grouping across all four cells.
   Plain omits the flag; explicit none canonicalizes away. Treatment sets rms.
   No other architecture feature is enabled, including PR74's input stem.
   Primary contrast is rms minus plain for each seed, not a cross-seed best.

6. **Validation and uncertainty.** Full official global_wp_clip2_v1 on1873
   sequences/3,528,273 scored rows. Paired whole-component bootstrap2000 draws,
   seed271828, conditional on fixed predictors. Report both seed deltas and
   their descriptive mean. One exposed development split and two seeds do not
   establish significance, training variance or public transfer. PR72 control
   grades .680881/.682194 are context, not reused control evidence.

7. **Leakage and representativeness.** Unchanged PR49 source split SHA256
   `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c`;
   train10325, valid fitting0, complete valid1873. Do not rebuild its index or
   change components. Chronological-schema naming is not proof of actual
   calendar chronology. Fit-only input statistics; normalization operates on
   the current row's head coordinates, never on sequence/batch/time statistics.
   No teacher/cache/initializer is used in fitting. Prefix, chunk/step/reset,
   cached-key, BF16/BF16x9 and two-rank gradient checks are prepared for receiver.

8. **Decision-changing result.** Mean rms−plain at least+.0005 WP with both
   paired deltas positive warrants a separately reviewed follow-up, not a
   launch or submission. Mixed/small/negative outcomes remain recorded. A
   confidence interval crossing zero alone does not close normalization as a
   direction. A failed cell stays incomplete until diagnosed; no automatic
   retries, extra seeds, alternative temperatures or endpoint selection.

9. **Execution.** Use RUNBOOK.md and `bash "$PACKET/run.sh" PHASE CELL` on an
   isolated clean Linux receiver at the reviewed full commit. Required env:
   WNN_EXPECTED_COMMIT,WNN_DATA_ROOT,WNN_SPLIT,WNN_RUN_ROOT,CUDA_VISIBLE_DEVICES.
   Stages: data-check → check → profile + profile-peak per cell → manual exact
   production-profile approval → fit → evaluate → bundle → compare. Outputs
   require persistent physical disk; /tmp,/dev/shm,tmpfs,ramfs are rejected.
   Fit limit8h/cell; each profile30min; DDP gate15min; evaluation2h. No commands
   were executed here. Optional evaluate-blend/compare-blend requires the exact
   frozen PR49 TL3 cache and all four fixed90/10 endpoints, with no tuning.

10. **Evidence and serving.** Return full config/source/split/runtime identity,
    test and two-rank receipts, profile/reload/memory evidence, reviewer approval,
    per-rank update/monitor logs, fixed-E4 predictions and canonical grades,
    actual model.pt/score/approved-profile SHA256 bindings, every paired delta,
    and all failure diagnostics. Standard bundles contain evidence plus final
    weights/resume state; preserve all-epoch weights/indexes on disk or in a
    companion archive. Large Transformer teacher weights are not a20MB CPU
    submission. SUBMISSION_STATUS.md records the current separate package audit.

11. **Results and publication.** New scientific results, GPU profiles and
    receiver acceptance are UNRUN. No new ZIP or public score is claimed.
    recipe.json freezes the panel; VALIDATION.json and INDEPENDENT_REVIEW.md
    record only static preparation. Append-only H/E ledgers and research/ideas
    indexes identify the packet. Implementation SHA: `67c351ae70cacef11f4ad485b5b98192f0d5e81c`.
    [Incremental code diff](https://github.com/VladKochetov007/WunderFunder/compare/4021de8ec5130143c207319c73a102a453eb911e...67c351ae70cacef11f4ad485b5b98192f0d5e81c); [GPU commands](https://github.com/VladKochetov007/WunderFunder/blob/codex/transformer-qk-normalization-20261109/research/experiment_drafts/transformer-qk-normalization-20261109/RUNBOOK.md).
    Merge and actual-execution SHAs remain null.

Static checks passed: Ruff0.16.7 lint/format over the full quality target,
training structure121 files/zero errors, AST/JSON/config hashes and symmetry,
Bash syntax and diff whitespace. Independent source review found no blocking
issue. Model/receiver execution remains UNRUN. The panel-bundle command retains
the aggregate paired comparison in addition to per-cell archives.

GitHub Training code quality passed on the implementation commit. Current
portal audit:108 team submissions,0/5 today,best.66832/rank45. No new
unsubmitted CPU-accepted ZIP was found; released checkpoint inputs and the
remaining export/CPU checks are pinned in SUBMISSION_INPUTS.json.
