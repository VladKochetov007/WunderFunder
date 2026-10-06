cd ~/WunderFunder-creative
while ! grep -q "^DONE" artifacts/logs/find256r_s4.log 2>/dev/null; do
  if grep -q Traceback artifacts/logs/find256r_s4.log; then echo FAILED; grep -A8 Traceback artifacts/logs/find256r_s4.log | tail -12; exit 1; fi; sleep 60; done
D=research/iterations/2026-10-02-slow-fast; cd $D
~/WunderFunder-creative/.venv/bin/python final_mix.py || exit 1
cd ~/WunderFunder-creative && CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/predict_run.py find256r_s4h2 valid --epoch 1 2>&1 | tail -1
cd $D && sed "s/pred_findm128_s3h2_valid/pred_find256r_s4h2_valid/" par_dm.py > par_raw256.py
rm -rf run_par256r && mkdir run_par256r && (cd run_par256r && unzip -q ../par256r.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python par_raw256.py run_par256r
rm -rf run_final_mix5 && mkdir run_final_mix5 && (cd run_final_mix5 && unzip -q ../final_mix5.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python time_pkg.py run_final_mix5
cat run_final_mix5/members.json; sha256sum final_mix5.zip
