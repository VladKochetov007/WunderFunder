#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
K=$ROOT/research/campaign_2026-10-26/kaggle_parent_numerics_r1
export PYTHONPATH=$ROOT/src WNN_THREADS=2 OMP_NUM_THREADS=2 WNN_LOADER_WORKERS=0 TMPDIR=/home/vlad/development/wunder/research_tmp
for spec in f0_s2001:0 f2_s2002:2; do cell=${spec%%:*}; fold=${spec#*:}
  for arm in fp16 fp16bf16ns; do r=kparent_${cell}_muon_${arm}_r1
    [ -f cache/pred_${r}_valid.npy ] || .venv/bin/python scripts/predict_run.py $r valid --epoch 2 --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold$fold.json --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $r > $K/export_$r.log 2>&1
  done
  .venv/bin/python - "$cell" "$fold" "$K" <<'PY'
import json,sys
cell,fold,K=sys.argv[1:]
arms={"fp32":{f"krepro_{cell}_muon_r3":1.0},"fp16":{f"kparent_{cell}_muon_fp16_r1":1.0},"fp16bf16ns":{f"kparent_{cell}_muon_fp16bf16ns_r1":1.0},"adamw_fp32":{f"krepro_{cell}_adamw_r3":1.0}}
json.dump({"folds":[int(fold)],"arms":arms,"comparisons":[["fp16","fp32"],["fp16bf16ns","fp32"],["fp16bf16ns","fp16"],["fp16bf16ns","adamw_fp32"]],"n_boot":2000},open(f"{K}/screen_{cell}.json","w"),indent=1)
PY
  PYTHONPATH=$ROOT/src:$ROOT/scripts .venv/bin/python scripts/screen_score.py $K/screen_${cell}.json --output $K/grade_${cell}.json > $K/score_${cell}.log 2>&1
done
echo done
