#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack; cd $ROOT
D=$ROOT/research/campaign_2026-10-26/diagnostics_r1
export PYTHONPATH=$ROOT/src WNN_THREADS=2 OMP_NUM_THREADS=2 WNN_LOADER_WORKERS=0 WNN_GPU_MEMORY_BYTES=1800000000 TMPDIR=/home/vlad/development/wunder/research_tmp
for r in swa345_muon6_f0_r1 swa345_adamw6_f0_r1; do
  [ -f cache/pred_${r}_valid.npy ] || .venv/bin/python scripts/predict_run.py $r valid --epoch 5 --campaign-split-manifest research/campaign_2026-09-29/artifacts/splits/fold0.json --campaign-full-valid-feature-only --device cuda --inference-precision fp32 --output-name $r > $D/export_$r.log 2>&1
done
cat > $D/screen_swa.json <<'J'
{"folds":[0],"arms":{"adamw":{"c3mix6_f0_s3_control_e6_r1":1.0},"muon":{"muon6_f0_s3_rnn_e6_r1":1.0},"adamw_swa":{"swa345_adamw6_f0_r1":1.0},"muon_swa":{"swa345_muon6_f0_r1":1.0}},
 "comparisons":[["adamw_swa","adamw"],["muon_swa","muon"],["muon_swa","adamw_swa"]],"n_boot":2000}
J
PYTHONPATH=$ROOT/src:$ROOT/scripts .venv/bin/python scripts/screen_score.py $D/screen_swa.json --output $D/grade_swa.json > $D/score_swa.log 2>&1
echo done
