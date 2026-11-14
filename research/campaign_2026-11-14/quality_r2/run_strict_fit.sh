#!/usr/bin/env bash
# Corrected FP32 optimizer factory, otherwise the frozen train-only recipe.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
S=/home/vlad/development/wunder/research_tmp/q14_strict_source
test ! -e "$O/LOCAL_GPU_HOLD"
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
name=$1; seed=$2; alpha=$3; shift 3
teacher=${WNN_Q14_TEACHER:-tt29rep_fbc1k_s5}
if [[ " $* " == *" --profile-steps "* ]]; then
  BASE=${BASE/--save-training-state/}
fi
test "$(git -C "$S" rev-parse HEAD)" = 48776fba8af26b16b7e6813def7ccf571f56c1ac
test -z "$(git -C "$S" status --porcelain)"
cd /home/vlad/development/wunder/research_tmp/chrono_root
export WNN_PROJECT_ROOT=$PWD PYTHONPATH=$S/src
export NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/home/vlad/development/wunder/research_tmp
test ! -e "artifacts/runs/$name/model_ep3.pt"
"$R/.venv/bin/python" -u "$R/research/campaign_2026-10-29/teacher_track_r1/strict_launch.py" \
  "$S" $BASE $PARENT --name "$name" --seed "$seed" \
  --distill "$teacher" --distill-alpha "$alpha" "$@" > "$O/$name.fit.log" 2>&1
echo "STRICT_FIT_DONE $name"
