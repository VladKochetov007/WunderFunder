cd ~/WunderFunder-creative
for f in 0 1; do g=$((5+f));
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh i0rb $f $g --i1swap 4557 --i1swap-noswap > /dev/null 2>&1 < /dev/null &
done
