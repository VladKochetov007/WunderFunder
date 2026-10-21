# B16 F1 offline artifact recovery

The B16 Kaggle worker completed its six-epoch seed-1004 fit and saved CP3/4/5, but stopped in the postfit helper before SWA derivation and prediction export. The helper required `initial_parameter_sha256`; the pinned clean trainer at `68ff682977b822f38351aac0f621146f5f13b440` did not emit that field. This is an artifact-schema incompatibility after fitting, not a failed fit or a quality result. The original terminal receipt and frozen R2 worker/grader remain unchanged.

This packet adds an offline recovery path. It verifies the clean trainer source hashes, cold-start arguments, no-init/no-resume/no-teacher ancestry, exact F1 fit/evaluation membership, six withheld epochs, normalization identity, 44,760 completed updates and saved CP3/4/5 hashes. It copies source artifacts into a new physical-disk root, then derives the exact equal-parameter CP3/4/5 mean using the pinned canonical `mean_state()` helper. The source checkpoints and source provenance are never modified.

The recovery receipt records `initial_parameter_sha256: null` and `initialization_identity_status: not_emitted_by_pinned_trainer_source`. The fit can be described as cold-start from its arguments, lack of a parent checkpoint and lack of a resume state; the initial parameter tensor digest remains unavailable. The worker receipt records FP32 arguments, `NVIDIA_TF32_OVERRIDE=0`, and disabled TF32 flags after profiling. The postfit failure means it did not record an after-fit runtime flag snapshot, so recovery does not claim that missing observation.

The collected source files are under `.../artifacts/artifacts/runs/r01_h512_f1_s1004_kaggle_fit_r2/`: CP3 is 8,904,103 bytes (`9bc7d517…b0899`), CP4 is 8,904,871 bytes (`d08c39fb…ffd14`), and CP5 is 8,905,703 bytes (`82cfa666…840a9`). The existing predictor inputs are `datasets/train.parquet`, `datasets/valid.parquet`, `cache/chains.npz`, and `cache/chain_folds.npz` in the project root. Recovery copies checkpoint outputs under separate `FIT` and `SWA` run names so the unmodified predictor can load each endpoint from its normal `artifacts/runs/<run>/model_ep5.pt` path.

## Bounded recovery commands

The command block below preserves the original full recovery sequence. Its
`derive` step has since completed and must not be repeated. For the current
local prediction-only path, use the [finite local fallback recipe](LOCAL_FALLBACK_R1.md),
which runs the remaining `verify-source` and `stage` checks before two serial
exports and the existing acceptance/grader.

Run only after the root has admitted the recovery within its separate two-GPU-hour allowance. These commands do not retrain. The derivation loads three small checkpoint files on CPU; staging creates symlinks to the existing raw data and two structural caches. The canonical predictor then writes two 457-group FP32 feature-only exports under the new recovery root. Use a detached, clean checkout of the exact producer commit with its pinned `pyproject.toml` and `uv.lock`; the current working branch's prediction code has changed since B16 ran.

```sh
set -euo pipefail
export WNN_LOADER_WORKERS=0 WNN_LOADER_RAM_GB=0.6 WNN_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 WNN_GPU_MEMORY_BYTES=4000000000
export TMPDIR=/home/vlad/development/wunder/research_tmp
PROJECT=/home/vlad/development/wunder/wnn_connectome_starterpack
INFERENCE_ROOT=/home/vlad/development/wunder/research_tmp/b16-inference-source-68ff
ARTIFACT_ROOT=/home/vlad/development/wunder/research_tmp/kaggle-h512-f1-b16-conditional-input-stage-20261021-r2/B16_TERMINAL_COLLECTION_V2_R3/outputs/results/h512_f1_b16_seed1004_profile_then_conditional_fit_r2/artifacts
RECOVERY_ROOT=/home/vlad/development/wunder/research_tmp/h512-f1-b16-offline-recovery-r1
DATA_ROOT="$PROJECT"
PACKET=research/campaign_2026-10-21/kaggle_h512_b16_offline_recovery_r1
FIT=r01_h512_f1_s1004_kaggle_fit_r2
SWA=r01_h512_f1_s1004_kaggle_swa345_r2
MANIFEST="$PROJECT/research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_profile_view.json"
REGISTRY="$PROJECT/research/campaign_2026-10-21/kaggle_h512_b16_conditional_f1_r2/fold1_portable_inputs.json"

git -C "$PROJECT" worktree add --detach "$INFERENCE_ROOT" 68ff682977b822f38351aac0f621146f5f13b440
uv run --project "$INFERENCE_ROOT" python "$PROJECT/$PACKET/recover_h512_f1_b16_offline.py" verify-source \
  --inference-source-root "$INFERENCE_ROOT"
uv run --project "$INFERENCE_ROOT" python "$PROJECT/$PACKET/recover_h512_f1_b16_offline.py" derive \
  --artifact-root "$ARTIFACT_ROOT" --recovery-root "$RECOVERY_ROOT" --project-root "$PROJECT"
uv run --project "$INFERENCE_ROOT" python "$PROJECT/$PACKET/recover_h512_f1_b16_offline.py" stage \
  --recovery-root "$RECOVERY_ROOT" --data-root "$DATA_ROOT"

PYTHONPATH="$INFERENCE_ROOT/src" WNN_PROJECT_ROOT="$RECOVERY_ROOT" WNN_PORTABLE_DATA_MANIFEST="$REGISTRY" \
  uv run --project "$INFERENCE_ROOT" python "$INFERENCE_ROOT/scripts/predict_run.py" "$FIT" valid --epoch 5 \
  --campaign-split-manifest "$MANIFEST" --campaign-evaluation-only \
  --output-name "$FIT" --inference-precision fp32
PYTHONPATH="$INFERENCE_ROOT/src" WNN_PROJECT_ROOT="$RECOVERY_ROOT" WNN_PORTABLE_DATA_MANIFEST="$REGISTRY" \
  uv run --project "$INFERENCE_ROOT" python "$INFERENCE_ROOT/scripts/predict_run.py" "$SWA" valid --epoch 5 \
  --campaign-split-manifest "$MANIFEST" --campaign-evaluation-only \
  --output-name "$SWA" --inference-precision fp32

WNN_PROJECT_ROOT="$RECOVERY_ROOT" WNN_PORTABLE_DATA_MANIFEST="$REGISTRY" \
  uv run --project "$INFERENCE_ROOT" python "$PROJECT/$PACKET/accept_h512_f1_b16_offline.py" \
  --artifact-root "$ARTIFACT_ROOT" --recovery-root "$RECOVERY_ROOT" \
  --inference-source-root "$INFERENCE_ROOT" --project-root "$PROJECT"
uv run --project "$INFERENCE_ROOT" python "$PROJECT/$PACKET/grade_h512_f1_b16_offline.py" \
  --artifact-root "$ARTIFACT_ROOT" --recovery-root "$RECOVERY_ROOT" \
  --inference-source-root "$INFERENCE_ROOT" --project-root "$PROJECT"
```

The acceptance and grade write create-only receipts under `RECOVERY_ROOT`. The new grader calls the frozen R2 `score_panel()` and pins the same five endpoints, coefficients, bootstrap seed, official metric, F1 memberships, target cache and scorer sources. It records a recovery-specific grade; it does not weaken or overwrite the frozen R2 grader or its original acceptance contract. The mixture rows are fixed output-space blends of the two prediction columns.

The F1 evaluation groups have prior exposure. Any resulting score is robustness evidence, not a clean holdout, public-score forecast or automatic promotion. The offline artifact derivation has since completed, as recorded in the recovery receipt. At local-fallback preparation on 2026-10-22, no prediction export or grade had run. A read-only run of `fit_audit.validate_fit_artifacts` against the collected metadata and checkpoint byte hashes passed: six withheld epochs, 44,760 updates, exact train/valid/evaluation membership, source-map match to 68ff and no emitted initialization digest. It did not load checkpoint tensors, target arrays or prediction arrays.

- [Finite B16 local prediction fallback](LOCAL_FALLBACK_R1.md) — prepares the two saved-checkpoint FP32 exports over the same 457 F1 evaluation groups, then invokes the frozen acceptance and grading path. Root admission is required after the six-fit panel; no GPU or scoring work ran during preparation.
