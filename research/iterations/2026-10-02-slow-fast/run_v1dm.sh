cd ~/WunderFunder-creative/research/iterations/2026-10-02-slow-fast
~/WunderFunder-creative/.venv/bin/python build_v1dm.py || exit 1
cd ~/WunderFunder-creative && CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/predict_run.py findm256_s3h2 valid --epoch 1 2>&1 | tail -1
cd ~/WunderFunder-creative/research/iterations/2026-10-02-slow-fast
sed "s/pred_findm128_s3h2_valid/pred_findm256_s3h2_valid/" par_dm.py > par_dm256.py
rm -rf run_par256 && mkdir run_par256 && (cd run_par256 && unzip -q ../par256.zip)
OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python par_dm256.py run_par256
for t in v1_dm v1_dm5; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python time_pkg.py run_$t; done
