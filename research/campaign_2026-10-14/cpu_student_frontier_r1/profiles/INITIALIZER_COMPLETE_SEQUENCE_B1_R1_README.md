# Complete-sequence initializer resource profile

The [contract](INITIALIZER_COMPLETE_SEQUENCE_B1_R1_CONTRACT.json) and [profile harness](../../../../scripts/profile_initializer_frontier.py) define a bounded actual-data CUDA smoke for the zero-state reference and three learned initializers (`constant`, `first_row`, `prefix`). The current [R3 CPU preflight](INITIALIZER_COMPLETE_SEQUENCE_B1_R3_PREFLIGHT.json) passed with an explicit 400 MB/one-CPU `scripts/limited` bound. It checked the frozen source tree, fitted control, cache contract/result, terminal B8 controller receipt, source-qualified split member, and target-file sizes. The execution entry point repeats these standard-library checks before importing PyTorch, then re-runs strict cache payload verification inside the admitted worker. The CPU preflight itself did not import PyTorch, rehash the large target arrays, allocate CUDA memory, update a model, or evaluate quality.

The profile uses the fitted F0 seed-3 H256×2 FP32 fixed-epoch-0 checkpoint (`model_ep0.pt`, SHA256 `1baa5835…7f006e6`) as a frozen backbone. The fit-validation receipt is SHA256 `1042e31a…fecdf43` and proves 14,920 completed updates, zero AMP skips, and finite recorded gradients. The fit producer source is pinned to `student-frontier-sequence-20261014` (`f0b5b6eb`); cache ordering and strict producer validation reuse the reviewed B8 profile helper. Runtime imports that fit’s trainer/model implementation and extends its `wnn` package path only to add the separate initializer modules.

The profiled sequence is the real epoch-0 sampler member `(valid, row_group=37, seq_ix=44,346,154)` at combined dataset position 10,562. Its numeric row-group ID also occurs in `train`; the bound loader preserves source-local cache rows (`valid` row 27, `train` row 36). The root B8 profile already verified both full homogeneous target payload hashes. The admitted worker repeats the unchanged strict cache validator and verifies all source-qualified mappings before building the B1 sample.

All modes use the same fitted backbone and 211-feature fitting-only mean/std buffers. Labels and teacher outputs are clipped to `[-2,2]`; the target is `0.1*label + 0.9*teacher`. The unchanged producer’s main propensity/true-mask weighting is applied to `abs(clipped label)`. The initializer helper masks rows 0–98 and divides one summed squared-error numerator by the total weight mass for the complete 20,000-row sequence. This global denominator and one update per complete sequence are explicit differences from the ordinary ten chunk-normalized, detached updates. The profile performs one FP32 AdamW update for each learned mode, with activation checkpointing across all ten 2,000-row chunks. It also checks finite loss/gradients and changed initializer parameters, bitwise-unchanged backbone tensors, zero-initialized prediction equivalence to zero state, full-versus-chunk output parity, explicit state reset, checkpoint reload, and finite contiguous FP32 native exports. This resource smoke does not fit a full recipe or make a quality claim.

The host reservation is 5 GB. The measured B8 terminal peak was 3.152 GB for strict cache validation and the full B8 sample; adding a conservative 100 MB initializer-specific CPU allowance gives a 3.252 GB working-set estimate and 1.748 GB headroom. CUDA activation/workspace demand is library-dependent, so no peak is extrapolated: the worker caps PyTorch allocation at 3.5 GB and refuses work at the 5 GB observed-VRAM ceiling. Root owns the GPU admission and launch. The profile has **not** been launched.

The frozen contract SHA256 is `588507017c4a63e12070f2018a7512d2fdde4ea8d2a8b76d51b81659fd71f703`; the harness SHA256 is `e8b43b167655e89005fe3ee6f11833e09efefaa1c7de92100eac24cd6d4c2cb0`; the current R3 preflight SHA256 is `8f729b44e80b64ccc8510cc2f07440cee91270891991d44345610154a7497863`. The [R1](INITIALIZER_COMPLETE_SEQUENCE_B1_R1_PREFLIGHT.json) and [R2](INITIALIZER_COMPLETE_SEQUENCE_B1_R2_PREFLIGHT.json) outputs are byte-identical predecessors tied to the earlier contract/source, SHA256 `76e167b6f50bccf02a094af569d1c795de82a614007fea2d4941de265e2cf193`. R3 is the sealed handoff preflight.

After a separate root admission, the exact controller command is:

```sh
MAIN=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/worktrees/student-initializer-profile-20261014
FIT=/home/vlad/development/wunder/worktrees/student-frontier-sequence-20261014
B8=/home/vlad/development/wunder/worktrees/student-frontier-b8-20261014
STATE="$MAIN/research/campaign_2026-10-06/bandit_continuation/runtime"
CONTRACT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R1_CONTRACT.json"
OUTPUT="$WT/research/campaign_2026-10-14/cpu_student_frontier_r1/profiles/INITIALIZER_COMPLETE_SEQUENCE_B1_R1_RESULT.json"

"$MAIN/scripts/bandit_resources.py" --root "$WT" --state "$STATE" \
  launch --attempt student-initializer-profile-f0s3-h256l2-b1-c2000-fp32-r1 \
  --kind gpu --cpus 1 --reserve-bytes 5000000000 \
  --profile 'one source-qualified clean F0 sequence; frozen H256x2 CP0; zero reference plus constant/first-row/prefix; one FP32 checkpointed full-sequence initializer update each; 3.5 GB allocator, 5 GB observed VRAM ceiling' \
  --contract "$CONTRACT" --python "$MAIN/.venv/bin/python" -- \
  timeout --signal=TERM --kill-after=10s 10m \
  /usr/bin/env WNN_PROJECT_ROOT="$MAIN" \
  PYTHONPATH="$WT/src:$FIT/src:$B8/src" \
  WNN_GPU_MEMORY_BYTES=3500000000 \
  "$MAIN/.venv/bin/python" "$WT/scripts/profile_initializer_frontier.py" \
  --data-root "$MAIN" --contract "$CONTRACT" --output "$OUTPUT" --execute
```
