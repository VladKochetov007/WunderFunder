cd ~/WunderFunder-creative
until [ $(grep -l "^DONE" artifacts/logs/fmuON_s3_h2fix.log artifacts/logs/fmuON_s4_h2fix.log 2>/dev/null | wc -l) -ge 2 ]; do
  grep -q Traceback artifacts/logs/fmuON_s?_h2fix.log && { echo FAILED; grep -A6 Traceback artifacts/logs/fmuON_s?.log | tail -12; exit 1; }; sleep 60; done
cd research/iterations/2026-10-02-big-ensemble
cat > build_on.py <<'PY'
exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_onorm",[f"{C}/fmuON_s3h2/model_ep1.pt",f"{C}/fmuON_s4h2/model_ep1.pt"])
pkg("par_onorm",[f"{C}/fmuON_s3h2/model_ep1.pt"])
PY
~/WunderFunder-creative/.venv/bin/python build_on.py 2>&1 | tail -2
cd ~/WunderFunder-creative && CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/predict_run.py fmuON_s3h2 valid --epoch 1 2>&1 | tail -1
cd research/iterations/2026-10-02-big-ensemble && rm -rf run_par_onorm && mkdir run_par_onorm && (cd run_par_onorm && unzip -q ../par_onorm.zip)
sed "s/pred_findm128_s3h2_valid/pred_fmuON_s3h2_valid/" ../2026-10-02-slow-fast/par_dm.py > par_on.py
OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python par_on.py run_par_onorm
rm -rf run_pair_onorm && mkdir run_pair_onorm && (cd run_pair_onorm && unzip -q ../pair_onorm.zip) && (cd run_pair_onorm && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3)
