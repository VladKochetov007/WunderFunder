# [Experiment] B300 causal raw-input stem screen: kernel 16 versus kernel 1

1. **Status, lineage and owner.** This is a local draft packet for six fresh
   fixed-4E B300 fits. No fit, profile, receiver gate, local test, inference, or
   remote job has run. Hypothesis H-B300-TRANSFORMER-CAUSAL-INPUT-STEM-20261109;
   experiment E-B300-TRANSFORMER-CAUSAL-INPUT-STEM-20261109. ROOT owns the
   source, receiver gates, and any later execution. The docs/config layer is
   prepared locally on branch codex/transformer-causal-input-stem-20261109.
   Parent PR72 head is 4021de8ec5130143c207319c73a102a453eb911e; intended
   target is research-team at 7e4074314a62d0907c9608f9220cf72958708a2d.
   This is stacked on PR72: a future PR to research-team includes inherited
   PR72 runtime changes. Review the incremental stem diff against the parent
   head separately from the inherited PR72 diff. This does not authorize
   publishing or merging PR72. The eventual execution commit, PR number, and
   merge SHA are unset.

2. **Why it should work.** A causal per-channel filter can expose short local
   changes to the temporal backbone while retaining the raw standardized row
   through an identity residual. Kernel 16 can use the preceding 15 rows and
   the current row; kernel 1 tests whether any effect is only an instantaneous
   per-channel factorization. The primary contrast is K16−K1, with K16−plain
   and K1−plain secondary. The strongest alternative is that any observed
   difference comes from the larger learned parameterization and optimizer
   interaction rather than useful lag structure. K1 is not parameter-count
   matched to K16. Causal prefix, chunk-carry, sequence-reset, zero-initialized
   weights, same per-seed backbone initialization, BF16/BF16x9, and full
   two-rank state-reload checks must pass before fitting.

   Convolutional sequence modeling is motivated by Bai et al.,
   [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for
   Sequence Modeling](https://arxiv.org/abs/1803.01271), and causal
   convolution/attention combinations by Verma,
   [Conformer LLMs](https://arxiv.org/abs/2307.00461). These papers motivate
   testing only; their tasks, architectures, and reported outcomes do not
   establish a gain here. The mechanism itself is not novel:
   [E-168](../../unified/archive/2026-10-12/ledger-snapshot/experiments.asof-20261012.jsonl#L168)
   tested a Conv16 encoder before a GRU on one exposed seed/fold and reset
   history at each chunk;
   [H-264](../../unified/archive/2026-10-12/ledger-snapshot/experiments.asof-20261012.jsonl#L620)
   had a negative fixed attention/TCN mixture at 10%;
   [large-dynamics](../../unified/ideas/LARGE_DYNAMICS_TRANSFORMER_B300_20261019_R2.md)
   already proposes causal convolution; and
   [PR35](https://github.com/VladKochetov007/WunderFunder/pull/35) implemented
   numeric/event features. This is a distinct Transformer input-stem contrast,
   not a novelty claim for convolution or numeric encoding.

3. **Compute advantage.** Six fits buy a three-arm, two-seed screen while
   holding the PR72 C0 recipe constant. Each fit is 4E × 820 updates/E =
   3,280 updates; all fits total 24 model-epochs and 19,680 optimizer updates.
   There are no new teacher or student fits and no reused controls. The
   receiver requires 12 exact-source/config/split engineering profiles
   (profile and zero-warmup profile-peak, each 50 updates per cell), or 600
   diagnostic updates. Old profile receipts are not reused. Each admitted fit
   uses two B300s. No GPU availability or reservation is claimed, and no
   wall-time estimate is made before profiles.

4. **Frozen comparison.** The six configs copy the PR72 C0
   Muon256/A4 config; only seed and model.input_stem_kernel differ. Plain omits
   the kernel field and retains the legacy no-stem path. Kernel 1 adds 112
   zero-initialized depthwise weights; kernel 16 adds 1,792. New stem weights
   use the AdamW-matrix group, outside the Muon block. No parameter-matched
   claim is made. Each seed uses a fresh backbone initialized identically
   across the three arms.

   Shared settings: Transformer hidden 768, 24 layers, 12 heads, FF multiplier
   4, context 1,024, dropout 0, hint dimension 0; chunk 256, gradient horizon
   256, accumulation 4, per-rank batch 128, world size 2, global sequence
   batch 256; Muon, LR 3e-4, weight decay 0.01, momentum 0.95, five
   orthogonalization steps, Muon epsilon 1e-7, AdamW betas 0.9/0.95 and
   epsilon 1e-8, neutral LR role/head/body factors 1, clip 1, 5% warmup,
   warmup_cosine_v2 and minimum LR ratio 0.1. Forward uses BF16/BF16x9 with
   FP32 master/optimizer arithmetic. Objective is cache-free
   standalone_labels_wmse_v2; normalization is all_fit and mask is
   all_after_warmup. Checkpoint endpoint is configured E4; all four epochs
   are retained for diagnostics only.

5. **Bounded fitting panel.** Exactly six new scientific fits:

   | Seed | Plain | K1 | K16 |
   |---|---|---|---|
   | 203201 | plain_s203201 | k1_s203201 | k16_s203201 |
   | 203202 | plain_s203202 | k1_s203202 | k16_s203202 |

   Each arm has two fits. There are zero reused controls, zero teacher fits,
   and zero student fits. Use the same physical two-GPU pair and runtime for
   all three arms within a seed, in plain → K1 → K16 order. The two seeds may
   use separate pairs. Profiles are engineering gates and do not count as
   fits. There is no automatic follow-up allocation.

6. **Validation and uncertainty.** Primary is the official
   global_wp_clip2_v1 on the full PR49 fold-0 validation set: 1,873 sequences
   and 3,528,273 scored rows. Compare same-seed predictions on identical rows
   and report each seed delta plus the descriptive two-seed mean. The
   component bootstrap uses 2,000 paired draws with RNG seed 271828 and is
   conditional on these trained fits; it summarizes evaluation-unit
   sensitivity, not training variance. This is one exposed fold and two seeds;
   it cannot establish significance, cross-fold robustness, or public transfer.

7. **Leakage and coverage.** Each input row uses only standardized raw112
   values. Kernel 16 sees no future values, pads only at sequence start, carries
   the preceding 15 standardized rows across chunks, and resets at sequence
   boundaries. Fit means and standard deviations use all_fit training
   membership only. The chronological split SHA-256 is
   0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c:
   10,325 training sequences fit the model and no validation sequence does.
   There is no KD, teacher, initializer, reference cache, or reused endpoint.
   Official validation remains exposed.

8. **Decision-changing result.** A later follow-up review is justified only if
   mean K16−K1 is at least +0.0005 WP, K16−K1 is positive for both seeds, and
   K16−plain is nonnegative for both seeds. These conditions allocate
   information to a separately reviewed proposal; they do not launch one.
   Mixed, small, negative, or failed outcomes remain recorded evidence about
   this exact recipe. An incomplete cell is preserved and investigated, with
   no automatic retry.

9. **How to run.** Receiver only: Linux, exact reviewed clean commit,
   canonical PR49 data/split paths, a fresh persistent physical run root, and
   exactly two allocated B300 IDs in CUDA_VISIBLE_DEVICES. Set
   WNN_EXPECTED_COMMIT, WNN_DATA_ROOT, WNN_SPLIT, WNN_RUN_ROOT, and
   CUDA_VISIBLE_DEVICES. The script sets CUBLAS_WORKSPACE_CONFIG=:4096:8 and
   CUBLAS_EMULATE_SINGLE_PRECISION=1 for checks, plus run-root scratch and
   compiler caches. It rejects non-full commit pins, dirty source, relative
   roots, and RAM-backed output mounts.

   From repository root, after ROOT authorizes receiver admission:

       PACKET=research/experiment_drafts/transformer-causal-input-stem-20261109
       bash "$PACKET/run.sh" data-check
       bash "$PACKET/run.sh" check

   Then for the verified pair assigned to seed 203201, profile each cell in
   order with both 50-update phases:

       bash "$PACKET/run.sh" profile plain_s203201
       bash "$PACKET/run.sh" profile-peak plain_s203201
       bash "$PACKET/run.sh" profile k1_s203201
       bash "$PACKET/run.sh" profile-peak k1_s203201
       bash "$PACKET/run.sh" profile k16_s203201
       bash "$PACKET/run.sh" profile-peak k16_s203201

   Repeat for seed 203202 on its verified pair. ROOT manually reviews the
   profile receipts and places the approved_profile.json production receipt
   for each cell before fit. Then use fit CELL, evaluate CELL, and bundle CELL
   for each frozen cell, followed by compare. Resume is only for a ROOT-reviewed
   safe continuation of that same cell. Optional evaluate-blend CELL and
   compare-blend require the exact PR49 reference cache and all six blend
   grades; they are outside the standalone primary screen.

   The packet is not authorized or launch-ready from this documentation
   alone. Profile, fit, evaluation, bundle, and comparison commands have not
   been run here.

10. **Evidence to return.** Preserve the exact source/config/split/runtime
    identities, data and mount receipts, module causality/reset/chunk tests,
    zero-init and same-backbone-init checks, BF16/BF16x9 and two-rank checks,
    all 12 profile logs/receipts, ROOT's per-cell approval receipts, per-rank
    and aggregate update records, epoch/LR/preclip norm/clip diagnostics,
    fixed-E4 predictions and official scores, checkpoint/score byte bindings,
    and a diagnosis for every failed or incomplete cell. Retain all four epoch
    weights on the host. The standard bundle contains the final model and
    resume state; keep the host run directory or a companion archive for
    intermediate weights and remaining evidence. Record archive SHA-256 and
    byte sizes, preserve the persistent run path, and record actual disk
    capacity and peak storage after admission; both remain unknown now.

11. **Results and trace.** No result from this screen exists: actual fits,
    profiles, gates, tests, inference, and grades are all zero. The branch is
    local, PR URL and merge SHA are null, and no public leaderboard change is
    claimed. Related PR69–73 evidence and its limits are summarized in the
    packet README. The append-only H/E ledgers and the root research,
    experiment-draft, ideas, reviews, and sources indexes link this packet.
    Current config hashes are in recipe.json; future run/archive hashes and
    source commit belong in the result trace after actual receiver completion.
