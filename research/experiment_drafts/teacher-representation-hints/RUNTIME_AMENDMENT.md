# Prospective runtime amendment — teacher hints (PR #29)

This amendment narrows the prior CP30 proposal to a runnable two-fold screen using the shared runtime. It does not authorize or start fitting. The original [CP30 PR text](PR_CP30_FROZEN.md) and [recipe](recipe.json) are preserved unchanged; the recipe remains historical and is not CLI input.

## Pinned runtime and exact configuration

Use [runtime PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32) at executable source commit [d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6](https://github.com/VladKochetov007/WunderFunder/commit/d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6). Its packet path is research/experiment_drafts/b300-transformer-distillation-runtime; the JSON amendment binds each config path and SHA256. The shared runtime supplies the raw112 trainer, causal sequence models, fitting-only caches, exporter and official scorer.

The four new configs are the pinned runtime's gru256_hint and gru256_permuted_hint files for F0/s1001 and F2/s1002. Each uses canonical GRU256×2, a 64-dimensional learned student hint projection, three epochs, batch size 1, 256-row chunks, four-chunk accumulation, AdamW at 3e-4, weight decay 0.01, clip 1.0 and 5% warmup. The aligned/permuted pair differs only in hint_mode. Both use output KD weight 1.0 and hint weight 0.1.

For each cell, reuse the exact same-fold T768×24 teacher/cache and the exact GRU256 output-KD endpoint from PR28. The teacher has 24 layers, width 768, 12 heads, 512-row sliding KV per layer, three epochs, strict FP32 and a frozen seed-1729 projection of its per-row-normalized final hidden state to 64 dimensions. Record and verify actual checkpoint, cache-manifest, source, split and fitting-member hashes before reuse; config identity alone does not establish artifact identity.

## Panel, cost and contrasts

| Cell | Reused O control | New H fit | New P fit | Teacher/cache |
|---|---|---|---|---|
| F0 / student seed 1001 / teacher seed 101001 | GRU256 output KD | aligned hints | component-permuted hints | reuse same-fold PR28 teacher/cache |
| F2 / student seed 1002 / teacher seed 101002 | GRU256 output KD | aligned hints | component-permuted hints | reuse same-fold PR28 teacher/cache |

Incremental cost: **4 new student fits**, **2 reused output-KD controls**, and **2 reused teachers/caches**. The control endpoints and teacher/cache artifacts must be independently verified; this packet does not invent their checkpoint hashes or storage paths. H−O asks whether aligned hints help beyond output KD; H−P asks whether row/component correspondence matters beyond adding the same hint loss. No new label-only or teacher fits are included.

Inputs are raw112, normalized using the first 32 fitting sequences only. Every full 20,000-row sequence is forwarded from row zero and reset at sequence boundaries. The loss/score mask is is_scored AND row_index >= 99. The student objective is additive: WMSE(clipped labels) + WMSE(clipped teacher outputs) + 0.1*MSE(hints); label and KD terms use abs(clipped label) weights. This replaces the historical alpha=0.9 label/teacher mixture. Student checkpoint selection is the configured terminal epoch, not outer-fold selection. Hints are training-only and are not student inference inputs.

## Explicit differences from the frozen proposal

The historical PR29 recipe specifies 90 student fits over CP30, three arms, an L207 teacher with anchored 4096-row recomputation, eight-epoch students, and an alpha=0.9 label/KD mixture. Its proposed teacher-hint projection and normalization also differ. Those terms remain frozen records only.

This amendment uses two exposed folds with one student seed per fold, four new fits, reused PR28 T768×24 teachers and output-KD controls, per-layer sliding KV, three fixed epochs, and additive WMSE/KD/hint losses. It is a small runnable screen, **not** a CP30 replication or confirmatory quality result. The old and new architectures, caches and objectives must not be combined under one experiment label.

## Receiving-host command

Run on the host that owns the approved B300 allocation, after PR32 remote acceptance. Fetch the published runtime PR and verify the exact source pin:

~~~bash
git fetch origin refs/pull/32/head:refs/remotes/origin/pr-32-runtime
git switch --detach d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6
test "$(git rev-parse HEAD)" = d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6
uv sync --frozen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test -n "$WNN_DATA_ROOT"
test -n "$WNN_SPLIT_F0"
test -n "$WNN_SPLIT_F2"
test -n "$WNN_RUN_ROOT"
test -n "$WNN_T768_F0_CACHE"
test -n "$WNN_T768_F2_CACHE"
test -n "$WNN_GRU256_KD_F0"
test -n "$WNN_GRU256_KD_F2"
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime
~~~

First complete the pinned runtime's tests, two-rank acceptance and actual-shape profile review. Profile each of the four configs on one GPU for eight updates, using the matching same-fold cache and a fresh profile directory. Review the receipt and create an approved copy only after the host checks pass. A representative F0 aligned profile command is:

~~~bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py \
  --config "$PACKET/configs/gru256_hint_f0_s1001.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --teacher-cache "$WNN_T768_F0_CACHE" \
  --output "$WNN_RUN_ROOT/pr29/f0_hint_profile" --profile-updates 8
~~~

Repeat with gru256_permuted_hint_f0_s1001.json, gru256_hint_f2_s1002.json, and gru256_permuted_hint_f2_s1002.json, each in its own profile directory and with the matching fold/cache. After approving each matching profile receipt, run the corresponding full fit with the same config/data/split/cache arguments, a fresh fit directory and --profile-receipt <profile-dir>/approved_profile.json. Reuse the PR28 O endpoints for paired reports; export each new endpoint with scripts/predict_sequence.py --membership evaluation and grade via scripts/score_sequence.py. Follow the common runtime README for exact export/score flags and result bundling.

## Status and evidence limits

No PR29 profile, fit, cache export, score, GPU acceptance or CPU-serving check is claimed. All four fits await cell ownership, canonical source/split bindings, verified PR28 artifact hashes and reviewed host profiles. Report each fold and both targets separately. With one seed per fold, this screen estimates neither across-seed uncertainty nor generalization to unseen folds; a component bootstrap over fixed predictions is sensitivity analysis only. A descriptive +0.001 WP signal can motivate follow-up, not a pass/fail gate. Mixed results, smaller effects or an interval crossing zero are not automatic rejection. Outer-fold scores remain exposed-fold reporting, not checkpoint selection or a public-score forecast.
