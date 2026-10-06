cd ~/WunderFunder-creative
until [ $(grep -l "^DONE" artifacts/logs/fmuQ256_s3.log artifacts/logs/fmuQ256_s4.log artifacts/logs/fmuQc500_s3.log 2>/dev/null | wc -l) -ge 3 ]; do
  grep -q Traceback artifacts/logs/fmuQ256_s?.log artifacts/logs/fmuQc500_s3.log 2>/dev/null && { echo FAILED; exit 1; }; sleep 60; done
D=research/iterations/2026-10-02-slow-fast; cd $D
sed -e 's/names=\["fmu256_s3h2","fmu256_s4h2","fmu128_s3h2","fmu128_s4h2","fmuc500_s3h2"\]/names=["fmuQ256_s3h2","fmuQ256_s4h2","fmuQ_s3h2","fmuQ_s4h2","fmuQc500_s3h2"]/' -e 's/pkg("par_mu",\[(f"{M}\/fmu128_s3h2.npz","none",1.0)\])//' -e 's/pkg("final_mu5"/pkg("final_muQ5"/' build_mu.py > build_muQ.py
grep -c fmuQ256 build_muQ.py
~/WunderFunder-creative/.venv/bin/python build_muQ.py || exit 1
rm -rf run_final_muQ5 && mkdir run_final_muQ5 && (cd run_final_muQ5 && unzip -q ../final_muQ5.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python time_pkg.py run_final_muQ5
sed -i "s/run_final_mu5/run_final_muQ5/" sumcheck.py && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python sumcheck.py
cat run_final_muQ5/members.json | grep name; sha256sum final_muQ5.zip
