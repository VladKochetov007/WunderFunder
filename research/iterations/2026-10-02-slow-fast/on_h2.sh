cd ~/WunderFunder-creative
for s in 3 4; do
 name=fmuON_s$s; log=artifacts/logs/${name}_h2fix.log
 ( export CUDA_VISIBLE_DEVICES=0 WNN_GPU_FRACTION=0.3 WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2 WNN_CPU_PCT=8 WNN_MEM_MAX=16G
   scripts/limited .venv/bin/python scripts/train_rnn.py --name ${name}h2 --train-source valid --include-heldout-valid --full-valid \
    --eval-source holdout --loss gcorr --mask-weight --batch 16 --epochs 2 --lr 5e-4 --aux-struct 0.3 --aux-next 0.3 \
    --hidden 128 --layers 2 --chunk 2000 --init ${name}@ep3 --freeze inp,gru --seed $s --fast --online-norm > $log 2>&1
   echo "DONE $(date -Is)" >> $log ) &
done
wait
tail -n 2 artifacts/logs/fmuON_s3_h2fix.log artifacts/logs/fmuON_s4_h2fix.log
sed -i 's/^until \[ \$(grep -l "^DONE" artifacts\/logs\/fmuON_s3.log artifacts\/logs\/fmuON_s4.log 2>\/dev\/null | wc -l) -ge 2 \]; do/until [ $(grep -l "^DONE" artifacts\/logs\/fmuON_s3_h2fix.log artifacts\/logs\/fmuON_s4_h2fix.log 2>\/dev\/null | wc -l) -ge 2 ]; do/; s/grep -q Traceback artifacts\/logs\/fmuON_s?.log/grep -q Traceback artifacts\/logs\/fmuON_s?_h2fix.log/' research/iterations/2026-10-02-slow-fast/on_wait.sh
bash research/iterations/2026-10-02-slow-fast/on_wait.sh
