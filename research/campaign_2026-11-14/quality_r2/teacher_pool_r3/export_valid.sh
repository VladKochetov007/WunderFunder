#!/usr/bin/env bash
# No fitting: compare the immutable imported team teacher endpoints.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
C=/home/vlad/development/wunder/research_tmp/chrono_root
test ! -e "$O/STOP"
cd "$C"
export WNN_PROJECT_ROOT=$C NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/home/vlad/development/wunder/research_tmp
for name in q14t_fbc_s5 q14t_tlead_s5 q14t_tlead_s6; do
  test ! -e "$O/STOP"
  test ! -e "cache/pred_${name}_valid.npy"
  "$R/.venv/bin/python" -u "$O/export_predictions_r2.py" "$name" valid --epoch 3 \
    --inference-precision fp32 \
    --campaign-split-manifest "$R/research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json" \
    --campaign-full-valid-feature-only > "$O/teacher_pool_r3/$name.valid.log" 2>&1
  echo "EXPORT_DONE $name"
done
