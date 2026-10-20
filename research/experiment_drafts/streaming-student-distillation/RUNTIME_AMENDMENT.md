# PR30 runtime amendment: compact streaming students

**Status: runtime implemented, panel unrun, B300 acceptance pending.** This is a
prospective amendment to PR30. It does not record a result or authorize a job.
The executable trainer and model are from [draft PR #32](https://github.com/VladKochetov007/WunderFunder/pull/32),
with implementation source SHA
`d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6`. The synchronized documentation/base
snapshot is `d7559e94`; the executable implementation remains pinned to the
source SHA above.

## Frozen panel and cost

Run two matched student objectives at each cell:

| Cell | Student seed | Label-only config | KD config | Exact PR28 reuse |
|---|---:|---|---|---|
| F0 | 1001 | `student_t128_labels_f0_s1001.json` | `student_t128_kd_f0_s1001.json` | F0 teacher/cache seed101001; GRU256 label and KD students seed1001 |
| F2 | 1002 | `student_t128_labels_f2_s1002.json` | `student_t128_kd_f2_s1002.json` | F2 teacher/cache seed101002; GRU256 label and KD students seed1002 |

This is **four new student fits**. PR28 contributes two exact teachers and
caches plus four exact GRU endpoints; these are prerequisites and reuse, not
new PR30 fits. Across PR28 and this amendment, the initial combined panel is
six PR28 fits plus four PR30 fits. Do not reuse any artifact based on filename
alone: bind source, split/component membership, configuration, seed, teacher
cache manifest, and endpoint hashes.

## Executable configuration

All four configs are JSON files under
`research/experiment_drafts/b300-transformer-distillation-runtime/configs/`.
They share the following values: schema version 1; raw 112-feature input;
Transformer width 128, two layers, four heads, FFN multiplier 4, dropout 0;
sliding context of 128 rows **per layer**; three epochs; batch size 1;
accumulate four 256-row chunks; seed 1001 at F0 and 1002 at F2; AdamW at
`3e-4`, weight decay `0.01`, gradient clip `1.0`, warmup fraction `0.05`, and
normalization from the first 32 fitting sequences. Checkpointing is enabled.
The context is per layer, not a 128-row total receptive field. No parameter
count or FLOP equivalence with GRU is claimed.

Both arms use the runtime's additive objective. Label-only uses
`label_WMSE + 0 * KD_WMSE`; KD uses `label_WMSE + 1.0 * KD_WMSE`. KD consumes the
same-row cached teacher outputs, while labels remain the student target. The
configs retain the learned 64-dimensional hint projection required by the
shared model schema, with hint weight zero in both arms; this is not a hint
experiment. The label-only arm does not receive a teacher cache. KD requires
the matching F0 or F2 PR28 cache.

All parameters, activations, states, gradients and optimizer arithmetic are
strict FP32; autocast, GradScaler, TF32 and quantization are off. Full sequence
rows, including warm-up, are forwarded causally, with scoring/training masks
excluding rows 0–98 and padding. Runtime state uses sliding per-layer KV
buffers. The terminal checkpoint is the fixed final epoch; there is no outer
validation selection.

## Schema change from historical CP30

| Field | Historical PR30 CP30 | Current executable amendment |
|---|---|---|
| Scope | Six arms × 30 paired cells = 180 student fits | Two arms × two cells = four student fits |
| Context | Student windows256/1024; teacher targets use an anchored prefix up to4096 rows | Student sliding KV128 per layer; shared teacher context512 per layer |
| KD objective | Alpha 0.9 mixture | Additive label WMSE plus KD WMSE at weight 1.0 |
| Fit horizon | Eight epochs, terminal epoch 7 | Three epochs, fixed final epoch 2 |
| Optimizer/update plan | Global sequence batch 32; update each 500-row core; AdamW `1e-3` | Batch 1; accumulate four 256-row chunks; AdamW `3e-4` |
| Runtime binding | Proposal with null source/runtime bindings | Shared runtime PR #32, pinned source SHA above; real data and B300 profile still unbound |

The historical `recipe.json` and original PR body are preserved unchanged. This
amendment changes only the current executable handoff; it does not rewrite past
specification or imply execution.

## Admission, commands, and outputs

The shared runtime's [remote acceptance checklist](https://github.com/VladKochetov007/WunderFunder/blob/d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6/research/experiment_drafts/b300-transformer-distillation-runtime/REMOTE_ACCEPTANCE.md)
and profile receipt are prerequisites. Start from an isolated checkout of the
pinned implementation and verify its commit before running anything:

```bash
git fetch origin d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6
: "${WNN_RUNTIME_CHECKOUT:?set an approved physical-disk checkout path}"
git worktree add --detach "$WNN_RUNTIME_CHECKOUT" d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6
cd "$WNN_RUNTIME_CHECKOUT"
git rev-parse HEAD  # must print d2e6263af3a0dcb2d35badec8c3fa15da0d91dc6
uv sync --frozen
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${WNN_DATA_ROOT:?bind the approved dataset/cache root}"
: "${WNN_SPLIT_F0:?bind the canonical F0 split manifest}"
: "${WNN_SPLIT_F2:?bind the canonical F2 split manifest}"
: "${WNN_RUN_ROOT:?bind a dedicated physical-disk output root}"
PACKET=research/experiment_drafts/b300-transformer-distillation-runtime
```

Run the matching real-shape engineering profile separately for each config,
using a fresh output directory and eight profile updates. Inspect the receipt,
remote acceptance evidence, memory headroom, and finite-gradient diagnostics;
then obtain recorded B300 admission and make an approved receipt copy without
changing its identity fields. A profile is discarded engineering work, not a
scientific fit or result. Example F0 label profile:

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py --config "$PACKET/configs/student_t128_labels_f0_s1001.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --output "$WNN_RUN_ROOT/pr30_f0_labels_profile" --profile-updates 8
```

For each KD profile also pass the matching PR28 cache via `--teacher-cache
"$WNN_RUN_ROOT/t768_f0_cache"` or `"$WNN_RUN_ROOT/t768_f2_cache"`; repeat the profile
with the corresponding config, split and fresh output directory for all four
config/cell combinations.

Only after all admission gates pass, fit the four cells with their matching
profile receipts and teacher-cache bindings:

```bash
uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py --config "$PACKET/configs/student_t128_labels_f0_s1001.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --output "$WNN_RUN_ROOT/pr30_f0_labels" \
  --profile-receipt "$WNN_RUN_ROOT/pr30_f0_labels_profile/approved_profile.json"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py --config "$PACKET/configs/student_t128_kd_f0_s1001.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F0" \
  --teacher-cache "$WNN_RUN_ROOT/t768_f0_cache" \
  --output "$WNN_RUN_ROOT/pr30_f0_kd" \
  --profile-receipt "$WNN_RUN_ROOT/pr30_f0_kd_profile/approved_profile.json"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py --config "$PACKET/configs/student_t128_labels_f2_s1002.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F2" \
  --output "$WNN_RUN_ROOT/pr30_f2_labels" \
  --profile-receipt "$WNN_RUN_ROOT/pr30_f2_labels_profile/approved_profile.json"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  scripts/train_sequence.py --config "$PACKET/configs/student_t128_kd_f2_s1002.json" \
  --data-root "$WNN_DATA_ROOT" --split-manifest "$WNN_SPLIT_F2" \
  --teacher-cache "$WNN_RUN_ROOT/t768_f2_cache" \
  --output "$WNN_RUN_ROOT/pr30_f2_kd" \
  --profile-receipt "$WNN_RUN_ROOT/pr30_f2_kd_profile/approved_profile.json"
```

Use the exact cache roots bound to the matching PR28 teacher/cell. Each fit
must have its own previously approved profile receipt. Keep checkpoints,
optimizer states, predictions and evidence archives outside Git; retain failed
attempts and identify each artifact with its hashes. Export, official scoring,
full-sequence parity, and CPU release admission follow the shared runtime and
remain separate downstream tasks.

## Interpretation and limits

The primary contrast is KD versus label-only for the same compact architecture,
cell, data order, seed and budget. Compare the KD Transformer with PR28's
GRU-KD endpoint only as a cross-architecture result; shapes, parameters and
FLOPs are not matched. A positive `+0.001 WP` is a useful signal for a bounded
follow-up, not an automatic promotion rule. Mixed signs, a smaller gain, or an
interval crossing zero do not automatically reject the candidate; interpret
both cells, both targets, uncertainty, compute and the exact matched controls.

No student fit, profile, score, B300 acceptance, CPU latency, package-size,
ZIP-readiness or public-score result is reported here. This research runtime
does not establish one-core/16-GB/60-minute deployment feasibility. Those gates
require an independently measured CPU path under the official contract.
