# Serial export and score commands, R1

This is a root-operated command manifest for the frozen [score plan](score_plan_r1.json) and [evaluation manifest](evaluation_manifest.json). It waits for both fixed CP1 fits to finish successfully, runs preflight, exports WMSE CP1, NLL CP1, and the audited Z4 CP1 in FP32, verifies all three artifacts, then runs the frozen seven-endpoint scorer. Nothing in this note was launched by the actor.

The preflight runs only after the fit report confirms both fixed CP1 fits and successful exits. It then binds each completed checkpoint and provenance hash plus exact fold fit membership before any export. It checks every explicitly supplied fit argument against the root's frozen `recipe.json`, binds initial-parent, propensity-checkpoint, and distillation-cache metadata/file hashes to the pinned parent and input-verification receipt, and verifies the valid Parquet's bytes, frozen `mtime_ns`, and footer against that receipt. The post-export receipt rechecks those hashes and adds sidecar identities and the full prediction byte hashes after sequence-at-a-time finiteness checks. Both receipts also bind all three validator module hashes and compare the complete producer code tree against the pinned commit. Only after the post-export receipt is written should scoring begin.

Run the block as one serial job inside root's admitted cgroup/systemd wrapper and shared GPU/cache-writer lock. The wrapper retains the plan limits: one CPU, 5,000,000,000 aggregate RAM bytes, 4,200,000,000 worker RAM bytes, 4,500,000,000 CUDA allocator bytes, zero swap, and loader workers zero. Use the already provisioned root environment and the physical-disk scratch path below. The pinned predictor rejects `--create-only` with `--campaign-full-valid-feature-only`; the preflight absence check and single-writer lock enforce the output gate. Keep the predictor argv exactly as pinned and do not run another cache writer concurrently.

Preflight and post-export audit verify that the producer worktree has no code diff from commit `68ff682977b822f38351aac0f621146f5f13b440` under `src/wnn`, `scripts`, or `utils.py`, and reject untracked Python files there. The plan's individual module pins remain in force. Preflight checks the frozen final grade path is absent along with new prediction and receipt outputs.

```bash
set -euo pipefail

ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
WT=/home/vlad/development/wunder/research_tmp/overnight-nll-adaptation-20261020
DIR=$WT/research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation
PY=$ROOT/.venv/bin/python
PLAN=$DIR/score_plan_r1.json
EVAL=$DIR/evaluation_manifest.json
AUDIT=$DIR/validate_prediction_artifacts.py
PREDICT=$WT/scripts/predict_run.py
SPLIT=$ROOT/research/campaign_2026-09-29/artifacts/splits/fold0.json
PREFLIGHT=$ROOT/research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/receipts/prediction_preflight_r1.json
POST=$ROOT/research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/receipts/prediction_post_export_r1.json
GRADE=$ROOT/research/campaign_2026-10-20/submission_delivery_r1/nll_adaptation/grades/f0_seven_endpoints_r1.json

export WNN_PROJECT_ROOT=$ROOT
export PYTHONPATH=$WT/src
export TMPDIR=$WT
export WNN_THREADS=1
export WNN_LOADER_WORKERS=0
export WNN_LOADER_RAM_GB=3.5
export WNN_GPU_MEMORY_BYTES=4500000000
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$(dirname "$PREFLIGHT")" "$(dirname "$GRADE")"

"$PY" "$AUDIT" --phase preflight --plan "$PLAN" --evaluation-manifest "$EVAL" --receipt "$PREFLIGHT" --post-receipt "$POST"

"$PY" "$PREDICT" overnight_nll_f0_s3_wmse_e2_r1 valid --epoch 1 --campaign-split-manifest "$SPLIT" --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name overnight_nll_f0_s3_wmse_e2_r1
"$PY" "$PREDICT" overnight_nll_f0_s3_nll_e2_r1 valid --epoch 1 --campaign-split-manifest "$SPLIT" --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name overnight_nll_f0_s3_nll_e2_r1
"$PY" "$PREDICT" oldz4_h230_student_f0_s3_r3 valid --epoch 1 --campaign-split-manifest "$SPLIT" --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --loader-workers 0 --output-name oldz4_h230_student_f0_s3_r3_fp32_reexport_r1

"$PY" "$AUDIT" --phase post-export --plan "$PLAN" --evaluation-manifest "$EVAL" --preflight-receipt "$PREFLIGHT" --receipt "$POST"

export PYTHONPATH=$ROOT/src:$ROOT/scripts
"$PY" "$ROOT/scripts/screen_score.py" "$DIR/score_screen_f0_r1.json" --output "$GRADE"
```

Each command exits nonzero on a failed gate; `set -e` stops the chain. The existing parent CP5 prediction is reused after its preflight byte, sidecar, checkpoint, and row-order checks. The historical Z4 FP16 cache is hashed as a protected reference and is never blended. The scorer follows the fixed screen definition: only 468 valid-source evaluation groups and 141 components enter the official score; the other 1,405 prediction rows remain fitting diagnostics. This F0 panel is exposed development evidence.
