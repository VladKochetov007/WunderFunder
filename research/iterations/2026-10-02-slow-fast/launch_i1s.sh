cd ~/WunderFunder-creative
for f in 0 1; do g=$((5+f));
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh i1sA $f $g --i1swap 4557 > /dev/null 2>&1 < /dev/null &
 WNN_GPU_FRACTION=0.45 setsid nohup scripts/fold_job.sh i1sB $f $g --i1swap 4557 --epoch-samples 12000 > /dev/null 2>&1 < /dev/null &
done
