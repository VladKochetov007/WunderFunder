cd ~/WunderFunder-creative
until [ $(grep -l "^DONE" artifacts/logs/ffull_s3.log artifacts/logs/ffull_s4.log artifacts/logs/fft4_s3.log artifacts/logs/fft4_s4.log artifacts/logs/fmuV2_s3.log artifacts/logs/fmuV2_s4.log 2>/dev/null | wc -l) -ge 6 ]; do
  grep -l Traceback artifacts/logs/{ffull,fft4,fmuV2}_s?.log 2>/dev/null && { echo FAILED; exit 1; }; sleep 60; done
cd research/iterations/2026-10-02-big-ensemble
cat > build_ftpairs.py <<'PY'
exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_fullft",[f"{C}/ffull_s3/model_ep1.pt",f"{C}/ffull_s4/model_ep1.pt"])
pkg("pair_ft4",[f"{C}/fft4_s3/model_ep3.pt",f"{C}/fft4_s4/model_ep3.pt"])
pkg("pair_v2",[f"{C}/fmuV2_s3h2/model_ep1.pt",f"{C}/fmuV2_s4h2/model_ep1.pt"])
PY
~/WunderFunder-creative/.venv/bin/python build_ftpairs.py 2>&1 | grep -E "^pair_[a-z0-9]+ [0-9a-f]{16}$"
for t in pair_fullft pair_ft4 pair_v2; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && echo $t && (cd run_$t && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3); done
