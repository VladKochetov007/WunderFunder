cd ~/WunderFunder-creative; D=research/iterations/2026-10-02-slow-fast
until [ $(grep -l "^DONE" artifacts/logs/fmuB256_s3.log artifacts/logs/fmuB256_s4.log 2>/dev/null | wc -l) -ge 2 ]; do
  grep -q Traceback artifacts/logs/fmuB256_s?.log 2>/dev/null && { echo FAILED; grep -A5 Traceback artifacts/logs/fmuB256_s?.log | tail; exit 1; }; sleep 120; done
sed -e "s/--hidden 128 --layers 2/--hidden 256 --layers 2/" -e "s/WNN_GPU_FRACTION=0.3/WNN_GPU_FRACTION=0.25/" $D/final_ft.sh > $D/final_ft256.sh; chmod +x $D/final_ft256.sh
for s in 3 4; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F", " '$2>3900{print $1; exit}'); g=${g:-0}
  bash $D/final_ft256.sh fbB_full_s$s fmuB256_s$s $s $g "--epochs 2 --lr 1e-4" &
  sleep 60
done; wait
cd $D/../2026-10-02-big-ensemble
cat > build_boot.py <<'PY'
exec(open("build_kf.py").read().split("R=H+")[0])
C=H+"/WunderFunder-creative/artifacts/runs"
pkg("pair_boot_h2",[f"{C}/fmuB256_s3h2/model_ep1.pt",f"{C}/fmuB256_s4h2/model_ep1.pt"])
pkg("pair_boot_fullft",[f"{C}/fbB_full_s3/model_ep1.pt",f"{C}/fbB_full_s4/model_ep1.pt"])
pkg("pair_boot_combo",[f"{C}/fmuB256_s3/model_ep3.pt",f"{C}/fmuB256_s3h2/model_ep1.pt",f"{C}/fmuB256_s4/model_ep3.pt",f"{C}/fmuB256_s4h2/model_ep1.pt"])
PY
~/WunderFunder-creative/.venv/bin/python build_boot.py 2>&1 | grep -E "^pair_boot_[a-z0-9]+ [0-9a-f]{16}$"
for t in pair_boot_h2 pair_boot_fullft pair_boot_combo; do rm -rf run_$t && mkdir run_$t && (cd run_$t && unzip -q ../$t.zip) && echo $t && (cd run_$t && OMP_NUM_THREADS=1 taskset -c 61 ~/venv311/bin/python ../check_e7.py | head -3); done
