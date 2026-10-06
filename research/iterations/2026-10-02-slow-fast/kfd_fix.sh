cd ~/WunderFunder-creative
S=research/iterations/2026-10-02-slow-fast/ft_sweep.sh
bash $S ft256d d256 2 5 "--freeze inp,gru --epochs 1 --lr 5e-4" > /dev/null 2>&1 < /dev/null &
bash $S ft256d d256 3 6 "--freeze inp,gru --epochs 1 --lr 5e-4" > /dev/null 2>&1 < /dev/null &
wait
tail -qn 1 artifacts/logs/ft256d_f2.log artifacts/logs/ft256d_f3.log
cd research/iterations/2026-10-02-big-ensemble
~/WunderFunder-creative/.venv/bin/python build_kfd.py 2>&1 | tail -7
for t in kf_d256 full_d256; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && (cd run_$t && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3); done
