#!/usr/bin/env bash
# Execute the reviewed four-fit dynamics-aux panel (E-FUTURE-INPUT-DYNAMICS-20261021-R1) on the
# corrected frozen source 781deb6a5, then the frozen paired grader. Recipes are unmodified except
# that relative script paths resolve into the frozen source tree.
set -euo pipefail
ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
S=/home/vlad/development/wunder/research_tmp/dynamics-screen-corrected-20261022
Q=$ROOT/research/campaign_2026-10-25/dynamics_panel_r1
PY=$ROOT/.venv/bin/python
cd $ROOT
export WNN_PROJECT_ROOT=$ROOT PYTHONPATH=$S/src TMPDIR=/home/vlad/development/wunder/research_tmp \
  WNN_THREADS=2 WNN_LOADER_WORKERS=0 WNN_GPU_MEMORY_BYTES=4500000000 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $Q/panel.log; }
for r in dynamics_f0_s1001_control_r1 dynamics_f0_s1001_aux005_r1 dynamics_f2_s1002_control_r1 dynamics_f2_s1002_aux005_r1; do
  plan=$S/research/experiment_drafts/dynamics-aux-20261021/recipes/$r.json
  n=$($PY -c "import json,sys;print(len(json.load(open(sys.argv[1]))['steps']))" $plan)
  for i in $(seq 0 $((n-1))); do
    mapfile -t A < <($PY -c "
import json,sys
s=json.load(open(sys.argv[1]))['steps'][int(sys.argv[2])]
print(s['name'])
for a in s['argv']:
    a=a.replace('{python}',sys.argv[3])
    if a.startswith('scripts/') and a.endswith('.py'): a=sys.argv[4]+'/'+a
    print(a)" $plan $i $PY $S)
    case "${A[0]}" in
      fit_*) [ -f $ROOT/artifacts/runs/$r/model_ep2.pt ] && { log "$r skip ${A[0]} (checkpoint exists)"; continue; } ;;
      export_*) [ -f $ROOT/cache/pred_${r}_valid.npy ] && { log "$r skip ${A[0]} (prediction exists)"; continue; } ;;
      score_*) A=("${A[0]}" "${A[@]:1}"); for j in "${!A[@]}"; do [[ "${A[$j]}" == $S/scripts/score_run.py ]] && A[$j]=$ROOT/scripts/score_run.py; done
               log "$r step ${A[0]} (main-checkout scorer)"
               PYTHONPATH=$ROOT/src:$ROOT/scripts "${A[@]:1}" > $Q/${r}_${A[0]}.log 2>&1 || { log "FAILED $r ${A[0]}"; exit 1; }
               continue ;;
    esac
    log "$r step ${A[0]}"
    "${A[@]:1}" > $Q/${r}_${A[0]}.log 2>&1 || { log "FAILED $r ${A[0]}"; exit 1; }
  done
done
log "grade panel"
PYTHONPATH=$ROOT/src:$ROOT/scripts $PY $Q/grader/grade_dynamics_panel.py \
  --output $Q/dynamics_panel_grade_R1.json > $Q/grade.log 2>&1 || log "grade failed"
log "panel done"
