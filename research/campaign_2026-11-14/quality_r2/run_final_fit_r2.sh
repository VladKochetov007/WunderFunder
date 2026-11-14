#!/usr/bin/env bash
# Corrected-source final-data refit, only after ENDPOINT_SELECTION_R2 admission.
set -euo pipefail
R=/home/vlad/development/wunder/wnn_connectome_starterpack
O=$R/research/campaign_2026-11-14/quality_r2
S=/home/vlad/development/wunder/research_tmp/q14_strict_source
test ! -e "$O/LOCAL_GPU_HOLD"
source "$R/research/campaign_2026-10-29/teacher_track_r1/recipe_args.sh"
name=$1; seed=$2; alpha=$3; boot=$4
test "$(git -C "$S" rev-parse HEAD)" = 48776fba8af26b16b7e6813def7ccf571f56c1ac
test -z "$(git -C "$S" status --porcelain)"
test ! -e "$O/STOP"
BASE=${BASE/--split-manifest $SPLIT/}
BASE=${BASE/--eval-source dev --full-valid/}
BASE=${BASE/--train-source train --train-seqs 10325/--train-source train+valid --include-heldout-valid --train-seqs 10607 --no-eval-during-fit}
cd /home/vlad/development/wunder/research_tmp/chrono_root
export WNN_PROJECT_ROOT=$PWD PYTHONPATH=$S/src
export NVIDIA_TF32_OVERRIDE=0 WNN_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TMPDIR=/home/vlad/development/wunder/research_tmp
test ! -e "artifacts/runs/$name/model_ep3.pt"
"$R/.venv/bin/python" -u "$R/research/campaign_2026-11-14/local_r1/final_launch.py" \
  "$S" $BASE $PARENT --name "$name" --seed "$seed" --distill tt29rep_fbc1k_s5 \
  --distill-alpha "$alpha" --distill-boot-h "$boot" > "$O/$name.fit.log" 2>&1
echo "STRICT_FINAL_DONE $name"
