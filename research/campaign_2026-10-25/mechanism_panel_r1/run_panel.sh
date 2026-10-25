#!/usr/bin/env bash
# i1 lead-block screen: 4 cold fits (control = zeroed block, lead = active block), exports, paired scoring.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
W=/home/vlad/development/wunder/research_tmp/mechanism-screen-r2-20261025
Q=$ROOT/research/campaign_2026-10-25/mechanism_panel_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT TMPDIR=/home/vlad/development/wunder/research_tmp WNN_THREADS=2 WNN_LOADER_WORKERS=0 \
  WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $Q/panel.log; }
while systemctl --user is-active -q wnn-clippanel; do sleep 60; done
for name in mech_f0_s1001_m1_muon_rmsmom_r1 mech_f0_s1001_e1_adamw_lr5e4_r1 mech_f0_s1001_e1b_muon_rms01_r1 mech_f0_s1001_m2_beta2_095_r1 mech_f0_s1001_m4_chrono_r1 mech_f0_s1001_tf_adamw_strict_fp32_r1 mech_f2_s1002_m1_muon_rmsmom_r1 mech_f2_s1002_e1_adamw_lr5e4_r1 mech_f2_s1002_e1b_muon_rms01_r1 mech_f2_s1002_m2_beta2_095_r1 mech_f2_s1002_m4_chrono_r1 mech_f2_s1002_tf_adamw_strict_fp32_r1; do
  fold=$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]]['fold'])" $Q/PLANS_R1.json $name)
  mapfile -t A < <($PY -c "import json,sys;[print(a) for a in json.load(open(sys.argv[1]))[sys.argv[2]]['argv']]" $Q/PLANS_R1.json $name)
  A=("${A[@]/#scripts\/train_rnn.py/$W/scripts/train_rnn.py}")
  if [ ! -f $ROOT/artifacts/runs/$name/model_ep2.pt ]; then log "fit $name"; PYTHONPATH=$W/src $PY "${A[@]}" > $Q/$name.fit.log 2>&1; fi
  if [ ! -f $ROOT/cache/pred_${name}_valid.npy ]; then log "export $name"
    PYTHONPATH=$W/src $PY $W/scripts/predict_run.py $name valid --epoch 2 \
      --campaign-split-manifest $ROOT/research/campaign_2026-09-29/artifacts/splits/fold$fold.json \
      --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $name > $Q/$name.export.log 2>&1; fi
done
for f in 0 2; do log "score F$f"; PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $ROOT/scripts/screen_score.py $Q/screen_f$f.json --output $Q/grade_f${f}_R1.json > $Q/score_f$f.log 2>&1 || log "score F$f failed"; done
log "panel done"
