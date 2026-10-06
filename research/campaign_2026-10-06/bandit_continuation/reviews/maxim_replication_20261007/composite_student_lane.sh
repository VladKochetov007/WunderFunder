#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/WunderFunder-creative"
seed=$1 gpu=$2
D="$PWD/research/iterations/2026-10-05-wave6"
export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=.65 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=6 WNN_MEM_MAX=32G
export WUNDER_RESEARCH_ROOT="$PWD" PYTHONPATH="$D/scripts:$HOME/WunderFunder/src"
for mode in direct distilled; do
 P=w18compcompress_${mode}_s${seed};log=artifacts/logs/$P.log
 test ! -e "artifacts/runs/$P"
 alpha=0; if [ "$mode" = distilled ]; then alpha=.7; fi
 scripts/limited .venv/bin/python "$D/scripts/train_rnn.py" --name "$P" --train-source train+valid --train-half 0 --valid-fold 0 --full-valid --eval-source vfold \
  --loss wmse --propensity prop_f0 --prop-power 2 --prop-eps .03 --valid-true-mask .03 --aux-struct .3 --aux-next .3 --hidden 256 --layers 2 --chunk 2000 --batch 16 \
  --distill w18comp_teacher --distill-alpha "$alpha" --family-teachers w15d256_s3,w6family_full,w6family_raw,orgbase \
  --epochs 2 --lr .0001 --init "w15d256_s${seed}@ep3" --seed "$seed" --fast > "$log" 2>&1
 for split in valid train; do
  extra=();if [ "$split" = train ]; then extra=(--groups-file "$PWD/research/iterations/2026-10-05-wave2/complement_hold.npy");fi
  scripts/limited .venv/bin/python "$D/scripts/predict_run.py" "$P" "$split" --epoch 1 "${extra[@]}" >> "$log" 2>&1
 done
 echo "DONE $(date -Is) $P" >> "$log"
done
