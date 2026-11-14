#!/usr/bin/env bash
# Sequential inference only, after both declared local fits finish.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
test ! -e "$O/LOCAL_GPU_HOLD"
C=/home/vlad/development/wunder/research_tmp/chrono_root
test ! -e "$O/STOP"
if systemctl --user is-active --quiet wnn-q14-fit-strict-local-pair.service; then
  echo 'Local fitting still active; export not admitted' >&2
  exit 2
fi
"$R/.venv/bin/python" "$O/verify_local_state.py" q14s_local_kd90_s3 q14s_local_kd100_s3
"$R/.venv/bin/python" "$O/trajectory_mean_r1/derive_mean.py"
cd "$C"
export WNN_PROJECT_ROOT=$C NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/vlad/development/wunder/research_tmp
for name in q14s_local_kd90_s3 q14s_local_kd100_s3 q14s_swa23_kd90_s3 q14s_swa23_kd100_s3; do
  test ! -e "$O/STOP"
  test ! -e "cache/pred_${name}_valid.npy"
  "$R/.venv/bin/python" -u "$O/export_predictions_r2.py" "$name" valid --epoch 3 \
    --inference-precision fp32 \
    --campaign-split-manifest "$R/research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json" \
    --campaign-full-valid-feature-only > "$O/$name.export.log" 2>&1
  echo "EXPORT_DONE $name"
done
