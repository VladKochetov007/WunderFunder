cd ~/WunderFunder-creative
while [ $(grep -l "^DONE" artifacts/logs/fmu256_s3.log artifacts/logs/fmu256_s4.log artifacts/logs/fmu128_s3.log artifacts/logs/fmu128_s4.log artifacts/logs/fmuc500_s3.log 2>/dev/null | wc -l) -lt 5 ]; do
  if grep -q Traceback artifacts/logs/fmu*.log; then echo FAILED; grep -l Traceback artifacts/logs/fmu*.log; exit 1; fi; sleep 60; done
D=research/iterations/2026-10-02-slow-fast; cd $D
~/WunderFunder-creative/.venv/bin/python build_mu.py || exit 1
cd ~/WunderFunder-creative && CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/predict_run.py fmu128_s3h2 valid --epoch 1 2>&1 | tail -1
cd $D && sed "s/pred_findm128_s3h2_valid/pred_fmu128_s3h2_valid/" par_dm.py > par_mu.py
rm -rf run_par_mu && mkdir run_par_mu && (cd run_par_mu && unzip -q ../par_mu.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python par_mu.py run_par_mu
rm -rf run_final_mu5 && mkdir run_final_mu5 && (cd run_final_mu5 && unzip -q ../final_mu5.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python time_pkg.py run_final_mu5
sed -i "s/run_final_mix5/run_final_mu5/" sumcheck.py && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python sumcheck.py
sha256sum final_mu5.zip
