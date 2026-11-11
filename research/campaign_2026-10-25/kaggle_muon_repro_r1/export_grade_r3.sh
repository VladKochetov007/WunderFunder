#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
K=$ROOT/research/campaign_2026-10-25/kaggle_muon_repro_r1
export PYTHONPATH=$ROOT/src WNN_THREADS=2 OMP_NUM_THREADS=2 WNN_LOADER_WORKERS=0 TMPDIR=/home/vlad/development/wunder/research_tmp
for spec in f0_s2001:0 f2_s2002:2; do cell=${spec%%:*}; fold=${spec#*:}
  for arm in adamw muon; do r=krepro_${cell}_${arm}_r3
    [ -f cache/pred_${r}_valid.npy ] || .venv/bin/python scripts/predict_run.py $r valid --epoch 2 --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold$fold.json --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $r > $K/export_$r.log 2>&1
  done
  .venv/bin/python - "$cell" "$fold" "$K" <<'PY'
import json,sys
cell,fold,K=sys.argv[1:]
json.dump({"folds":[int(fold)],"arms":{"adamw":{f"krepro_{cell}_adamw_r3":1.0},"muon":{f"krepro_{cell}_muon_r3":1.0}},"comparisons":[["muon","adamw"]],"n_boot":2000,"class_mix":{"public_fit_E298":[0.0,0.525,0.05,0.425]}},open(f"{K}/screen_r3_{cell}.json","w"),indent=1)
PY
  PYTHONPATH=$ROOT/src:$ROOT/scripts .venv/bin/python scripts/screen_score.py $K/screen_r3_${cell}.json --output $K/grade_r3_${cell}.json > $K/score_r3_${cell}.log 2>&1
done
echo done
