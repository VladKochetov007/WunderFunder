#!/usr/bin/env bash
# One fixed-recipe fit. Resource bounds belong to its systemd unit.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
name=$1; seed=$2; shift 2
cd /home/vlad/development/wunder/research_tmp/chrono_root
export WNN_PROJECT_ROOT=$PWD PYTHONPATH=/home/vlad/development/wunder/research_tmp/pr49_wt/src
export NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/home/vlad/development/wunder/research_tmp
test ! -e "artifacts/runs/$name/model_ep3.pt"
"$R/.venv/bin/python" -u "$R/research/campaign_2026-10-29/teacher_track_r1/strict_launch.py" \
  /home/vlad/development/wunder/research_tmp/pr49_wt $BASE $PARENT \
  --name "$name" --seed "$seed" --distill tt29rep_fbc1k_s5 --distill-alpha .9 "$@" \
  > "$O/$name.fit.log" 2>&1
echo "FIT_DONE $name"
