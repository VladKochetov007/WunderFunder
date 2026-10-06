cd ~/WunderFunder-creative
while [ $(grep -l "^DONE" artifacts/logs/kfd256_f2.log artifacts/logs/kfd256_f3.log 2>/dev/null | wc -l) -lt 2 ]; do
  if grep -q Traceback artifacts/logs/kfd256_f?.log; then echo FAILED; grep -B2 -A6 Traceback artifacts/logs/kfd256_f?.log | tail -20; exit 1; fi; sleep 60; done
ls artifacts/runs/ft256d_f2a artifacts/runs/ft256d_f3a >/dev/null || { echo missing FT; exit 1; }
cd research/iterations/2026-10-02-big-ensemble
cat > build_kfd.py <<'PY'
exec(open("build_kf.py").read().split("R=H+")[0])
O=H+"/WunderFunder/artifacts/runs";C=H+"/WunderFunder-creative/artifacts/runs"
pkg("kf_d256",[f"{O}/ft256d_f0h/model.pt",f"{O}/ft256d_f1h/model.pt",f"{C}/ft256d_f2a/model.pt",f"{C}/ft256d_f3a/model.pt"])
pkg("full_d256",[f"{O}/find256_s3h/model.pt"])
PY
~/WunderFunder-creative/.venv/bin/python build_kfd.py 2>&1 | tail -7
for t in kf_d256 full_d256; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && (cd run_$t && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3); done
