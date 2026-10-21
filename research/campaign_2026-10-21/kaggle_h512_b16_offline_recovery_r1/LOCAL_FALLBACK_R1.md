# B16 local F1 prediction fallback R1

This is a finite local prediction-only recovery for the saved B16 fit. The
[Kaggle export terminal receipt](../next_cycle_r1/B16_CLOUD_TERMINAL_COLLECTION_R1.json)
records a 700-second timeout, no prediction
artifacts, zero observed GPU allocation, and no SWA export start. Its diagnosis
that portable raw-input hashing may have occupied the worker is explicitly
unconfirmed. The provider receipt records 709.803 seconds consumed and
1,635.575 seconds remaining; root has closed further cloud launches.

The offline artifact-recovery receipt already exists at
`/home/vlad/development/wunder/research_tmp/h512-f1-b16-offline-recovery-r1/B16_OFFLINE_RECOVERY_RECEIPT_R1.json`.
It records `OFFLINE_ARTIFACT_RECOVERY_COMPLETE_NOT_GRADED`, source commit
`68ff682977b822f38351aac0f621146f5f13b440`, and no training run or quality
evaluation. CP5 and the equal-parameter CP3/4/5 mean (`SWA345`) are already
derived there. At the 2026-10-22 path check, the prediction-root stage receipt
and both prediction outputs were absent. This recipe reuses the existing
recovery; it does not re-copy checkpoints or re-derive SWA.

The only model work is two serial exports from those saved checkpoints:

| Endpoint | Run name | Checkpoint | Evaluation membership |
|---|---|---|---|
| CP5 | `r01_h512_f1_s1004_kaggle_fit_r2` | epoch 5 | frozen 457-group F1 evaluation set |
| SWA345 | `r01_h512_f1_s1004_kaggle_swa345_r2` | epoch 5 | the same frozen 457-group F1 evaluation set |

The canonical predictor is an exact clean checkout of commit
`68ff682977b822f38351aac0f621146f5f13b440`. Both calls use the frozen F1
manifest with `--campaign-evaluation-only`, so they read features for the
evaluation groups and do not load targets. `--inference-precision fp32` forces
the common FP32 evaluation path; `--device cuda` and `NVIDIA_TF32_OVERRIDE=0`
preserve the fixed precision requirements. The existing acceptance script
validates both exports before the unchanged five-endpoint F1 grader reads the
official targets and scores them. No fitting, teacher work, new coefficients,
source change, or score-plan change is part of this recovery.

Root owns admission. Start only after the current six-fit panel has released
the local GPU and root has confirmed the local resource envelope. Run each
command once, serially, under one CPU (100% quota), a 3,000,000,000-byte RAM
limit, zero task swap, one visible GPU, a 4,000,000,000-byte PyTorch allocator
limit, and a 5,000,000,000-byte VRAM stop threshold. The cumulative wall cap
for both prediction calls is 7,200 seconds.
Do not start a waiter, queue, retry, or parallel export. Stop if the device,
memory headroom, or remaining wall budget fails admission.

The 3 GB host limit is a proposed admission ceiling, not a measured result for
this prediction path. Its static basis is synchronous single-process loading
(`workers=0`), batch size 16 over 112 raw FP32 features (143,360,000 bytes per
raw feature batch), and a disk-backed FP32 output memmap. One endpoint's payload is
`457 * 20,000 * 2 * 4 = 73,120,000` bytes; endpoints are written serially. The
fixed grader opens prediction arrays memory-mapped and scores one sequence at
a time. Staging and predictor preflight still hash/check the supplied inputs;
the two raw Parquet inputs total 34,855,728,139 bytes. Include this I/O and
hashing work inside the wall cap. Do not skip the portable registry check.

## Exact command sequence

Run this only after root's explicit local resource admission. It uses the
already provisioned root Python environment; it does not install or sync
dependencies. The source worktree must be new and clean. If its path already
exists, stop and verify its identity instead of replacing it.

```sh
set -euo pipefail
PROJECT=/home/vlad/development/wunder/wnn_connectome_starterpack
PYTHON="$PROJECT/.venv/bin/python"
INFERENCE_ROOT=/home/vlad/development/wunder/research_tmp/b16-inference-source-68ff
ARTIFACT_ROOT=/home/vlad/development/wunder/research_tmp/kaggle-h512-f1-b16-conditional-input-stage-20261021-r2/B16_TERMINAL_COLLECTION_V2_R3/outputs/results/h512_f1_b16_seed1004_profile_then_conditional_fit_r2/artifacts
RECOVERY_ROOT=/home/vlad/development/wunder/research_tmp/h512-f1-b16-offline-recovery-r1
PACKET=research/campaign_2026-10-21/kaggle_h512_b16_offline_recovery_r1
MANIFEST="$PROJECT/research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_profile_view.json"
REGISTRY="$PROJECT/research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_portable_inputs.json"
FIT=r01_h512_f1_s1004_kaggle_fit_r2
SWA=r01_h512_f1_s1004_kaggle_swa345_r2

export WNN_LOADER_WORKERS=0 WNN_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 WNN_GPU_MEMORY_BYTES=4000000000
export TMPDIR=/home/vlad/development/wunder/research_tmp

test ! -e "$INFERENCE_ROOT" && test ! -L "$INFERENCE_ROOT"
git -C "$PROJECT" worktree add --detach "$INFERENCE_ROOT" 68ff682977b822f38351aac0f621146f5f13b440
"$PYTHON" "$PROJECT/$PACKET/recover_h512_f1_b16_offline.py" verify-source \
  --inference-source-root "$INFERENCE_ROOT"

test -f "$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_RECEIPT_R1.json"
test ! -e "$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_STAGE_R1.json"
test ! -e "$RECOVERY_ROOT/datasets" && test ! -L "$RECOVERY_ROOT/datasets"
test ! -e "$RECOVERY_ROOT/cache" && test ! -L "$RECOVERY_ROOT/cache"
"$PYTHON" "$PROJECT/$PACKET/recover_h512_f1_b16_offline.py" stage \
  --recovery-root "$RECOVERY_ROOT" --data-root "$PROJECT"

# The pinned evaluation-only path rejects --create-only and --output-dir.
# Preflight every direct-write destination, including partial and stale files.
for path in \
  "$RECOVERY_ROOT/cache/pred_${FIT}_valid.npy" \
  "$RECOVERY_ROOT/cache/pred_${FIT}_valid.npy.partial" \
  "$RECOVERY_ROOT/cache/pred_${FIT}_valid.json" \
  "$RECOVERY_ROOT/cache/seqsums_${FIT}.npz" \
  "$RECOVERY_ROOT/cache/pred_${SWA}_valid.npy" \
  "$RECOVERY_ROOT/cache/pred_${SWA}_valid.npy.partial" \
  "$RECOVERY_ROOT/cache/pred_${SWA}_valid.json" \
  "$RECOVERY_ROOT/cache/seqsums_${SWA}.npz" \
  "$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_ACCEPTANCE_R1.json" \
  "$RECOVERY_ROOT/F1_FIXED_SCREEN_GRADE_OFFLINE_R1.json"
do
  if test -e "$path" || test -L "$path"; then
    printf 'Refusing non-create-only prediction/grade output: %s\n' "$path" >&2
    exit 1
  fi
done

export PYTHONPATH="$INFERENCE_ROOT/src"
export WNN_PROJECT_ROOT="$RECOVERY_ROOT"
export WNN_PORTABLE_DATA_MANIFEST="$REGISTRY"
"$PYTHON" "$INFERENCE_ROOT/scripts/predict_run.py" "$FIT" valid --epoch 5 \
  --campaign-split-manifest "$MANIFEST" --campaign-evaluation-only \
  --output-name "$FIT" --device cuda --inference-precision fp32 --loader-workers 0
"$PYTHON" "$INFERENCE_ROOT/scripts/predict_run.py" "$SWA" valid --epoch 5 \
  --campaign-split-manifest "$MANIFEST" --campaign-evaluation-only \
  --output-name "$SWA" --device cuda --inference-precision fp32 --loader-workers 0

"$PYTHON" "$PROJECT/$PACKET/accept_h512_f1_b16_offline.py" \
  --artifact-root "$ARTIFACT_ROOT" --recovery-root "$RECOVERY_ROOT" \
  --inference-source-root "$INFERENCE_ROOT" --project-root "$PROJECT"
"$PYTHON" "$PROJECT/$PACKET/grade_h512_f1_b16_offline.py" \
  --artifact-root "$ARTIFACT_ROOT" --recovery-root "$RECOVERY_ROOT" \
  --inference-source-root "$INFERENCE_ROOT" --project-root "$PROJECT"
```

`--create-only` cannot be used for these outputs: the pinned predictor rejects
it together with `--campaign-evaluation-only`. It also rejects an explicit
`--output-dir` unless create-only is enabled. The path preflight above is
therefore the write-once guard for the unmodified evaluation-only predictor;
if any listed file exists, stop rather than overwrite or retry. The predictor
also removes a stale `seqsums_<run>.npz`, so those paths are included.

## Receipts and review gates

The existing artifact-recovery receipt is
`$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_RECEIPT_R1.json`. The stage command creates
`$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_STAGE_R1.json`. The two predictor calls
create `cache/pred_<run>_valid.npy` and matching `.json` sidecars in that root.
Acceptance creates
`$RECOVERY_ROOT/B16_OFFLINE_RECOVERY_ACCEPTANCE_R1.json`; the grader creates
`$RECOVERY_ROOT/F1_FIXED_SCREEN_GRADE_OFFLINE_R1.json`. These are new, separate
recovery outputs; frozen cloud receipts, fit artifacts, and the original
grader remain unchanged.

The source and consumer checks are bounded and source-pinned:

- `verify-source` requires the exact clean 68ff checkout and pinned
  `pyproject.toml`, lockfile, predictor and trainer-entrypoint hashes.
- `stage` checks the recovery receipt, exact raw input sizes, pinned structural
  cache hashes and portable-registry hash before recording the staged root.
  The registry performs the full raw-file identity check during predictor
  preflight; its result is not waived because the cloud phase was unclear.
- The predictor uses the frozen manifest's 457 evaluation groups, writes
  feature-only FP32 arrays, and records checkpoint, membership, source and
  normalization identities in sidecars. It does not read official target
  values on this path.
- Acceptance checks the recovered checkpoints, SWA derivation, exact source,
  NPY header/shape/dtype, full row coverage/order, precision, feature-only
  metadata and sidecar hashes before the grader is called.
- The unchanged grader checks its pinned score plan and targets, memory-maps
  prediction files, and evaluates the fixed five-endpoint F1 panel under the
  official scorer. The F1 evaluation membership is previously exposed and is
  robustness evidence, not a clean holdout or public-score forecast.

No local prediction, acceptance or grade was run while preparing this recipe.
