#!/usr/bin/env bash
# Valid-adaptation sweep on fold parents: FT on valid chains outside the held-out fold, predict all valid.
#   ft_sweep.sh NAME PARENT FOLD GPU STAGE1_FLAGS [-- STAGE2_FLAGS]
# STAGE flags are extra train_rnn args (e.g. "--freeze inp,gru --epochs 2 --lr 5e-4"). Output cache/pred_NAME_fFOLD_valid.npy.
set -euo pipefail
cd ~/WunderFunder-creative
name=$1 parent=$2 fold=$3 gpu=$4 s1=$5 s2=${6:-}
out="${name}_f${fold}"; log="artifacts/logs/${out}.log"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=0.3 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=8 WNN_MEM_MAX=16G
common=(--train-source valid --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --aux-struct 0.3 --aux-next 0.3
        --hidden 128 --layers 2 --seed 3 --fast --valid-fold $fold --eval-source vfold)
echo "START $(date -Is) $out parent=${parent}_f${fold} s1=[$s1] s2=[$s2]" >> $log
scripts/limited .venv/bin/python scripts/train_rnn.py --name ${out}a --init ${parent}_f${fold}@ep3 "${common[@]}" $s1 >> $log 2>&1
last=${out}a
if [ -n "$s2" ]; then
  scripts/limited .venv/bin/python scripts/train_rnn.py --name ${out}b --init ${out}a "${common[@]}" $s2 >> $log 2>&1
  last=${out}b
fi
scripts/limited .venv/bin/python scripts/predict_run.py $last valid --output-name ${out} >> $log 2>&1
echo "DONE $(date -Is) $out" >> $log
