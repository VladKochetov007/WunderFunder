cd ~/WunderFunder-creative
until [ $(grep -l "^DONE" artifacts/logs/fmu{P1,S0,V0}_s{3,4}.log 2>/dev/null | wc -l) -ge 6 ]; do
  grep -q Traceback artifacts/logs/fmu{P1,S0,V0}_s{3,4}.log 2>/dev/null && { echo FAILED; grep -l Traceback artifacts/logs/fmu{P1,S0,V0}_s{3,4}.log; exit 1; }; sleep 60; done
cd research/iterations/2026-10-02-big-ensemble
cat > build_pairs2.py <<'PY'
exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
for tag,pre in (("pair_prop1","fmuP1"),("pair_since0","fmuS0"),("pair_vtm0","fmuV0")):
    pkg(tag,[f"{C}/{pre}_s3h2/model_ep1.pt",f"{C}/{pre}_s4h2/model_ep1.pt"])
PY
~/WunderFunder-creative/.venv/bin/python build_pairs2.py 2>&1 | grep -v "^pair_.*fmu" 
for t in pair_prop1 pair_since0 pair_vtm0; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && echo $t && (cd run_$t && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3); done
