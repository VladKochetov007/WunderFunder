#!/usr/bin/env bash
# After the local GPU frees: export Kaggle NorMuon/TabM8 runs and TabM8 members, then grade vs R3 Muon.
set -uo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
D=$ROOT/research/campaign_2026-10-29/tabm_gru_r1
while systemctl --user is-active -q wnn-tabm29-local; do sleep 30; done
export NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$ROOT/src WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=300% WNN_THREADS=3 WNN_MEM_MAX=10000000000
pred() { [ -f cache/pred_$1_valid.npy ] || scripts/limited .venv/bin/python scripts/predict_run.py $1 valid --epoch 2 --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold$2.json --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $1 > $D/$1.export.log 2>&1 || echo "EXPORT_FAIL $1"; }
for c in f0_s2001:0 f2_s2002:2; do cell=${c%%:*}; fold=${c#*:}
  pred tabm29_${cell}_normuon_r1 $fold
  pred tabm29_${cell}_tabm8_r1 $fold
  [ -d artifacts/runs/tabm29_${cell}_tabm8_r1_m7 ] || .venv/bin/python $D/split_members.py tabm29_${cell}_tabm8_r1 2 > /dev/null
  for m in 0 1 2 3 4 5 6 7; do cp -n artifacts/runs/tabm29_${cell}_tabm8_r1/provenance.json artifacts/runs/tabm29_${cell}_tabm8_r1_m$m/; pred tabm29_${cell}_tabm8_r1_m$m $fold; done
  .venv/bin/python - "$cell" "$fold" "$D" <<'PY'
import json,sys
cell,fold,D=sys.argv[1:]
c=f"krepro_{cell}_muon_r3"
json.dump({"folds":[int(fold)],"arms":{"muon":{c:1.0},"tabm8":{f"tabm29_{cell}_tabm8_r1":1.0},"normuon":{f"tabm29_{cell}_normuon_r1":1.0}},
 "comparisons":[["tabm8","muon"],["normuon","muon"]],"n_boot":2000,"class_mix":{"public_fit_E298":[0.0,0.525,0.05,0.425]}},open(f"{D}/screen_kaggle_{cell}.json","w"),indent=1)
PY
  PYTHONPATH=$ROOT/src:$ROOT/scripts scripts/limited .venv/bin/python scripts/screen_score.py $D/screen_kaggle_${cell}.json --output $D/grade_kaggle_${cell}.json > $D/score_kaggle_${cell}.log 2>&1
done
echo KAGGLE_GRADE_DONE
